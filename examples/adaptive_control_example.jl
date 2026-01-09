using ComponentArrays
using ControlSystems
using DifferentialEquations
using UnPack
using Plots

# ## Helper functions
# First, we need a way to apply inputs to the system through keyword arguments. These
# will help us pass in inputs as either values or functions of (x,p,t).
maybe_apply(f::Function, x, p, t) = f(x, p, t)
maybe_apply(f, x, p, t) = f

function apply_inputs(func; kwargs...)
    simfun(dx, x, p, t) = func(
        dx, x, p, t; map(f -> maybe_apply(f, x, p, t), (; kwargs...))...
    )
    simfun(x, p, t) = func(x, p, t; map(f -> maybe_apply(f, x, p, t), (; kwargs...))...)
    return simfun
end

# Next, we need a way to create derivative functions from transfer functions. In ControlSystems.jl
# there is a function called `simulator` that does this, but the inputs must be applied from
# the start, so we couldn't use it as a component function. Our version allows inputs to be
# passed through the keyword arguments and, as an added convenience, is in observer canonical
# form so our first element of `x` is also the output `y` (note that while this is true for
# our problem, it isn't always going to be the case).
SISO_simulator(P::TransferFunction) = SISO_simulator(ss(P))
function SISO_simulator(P::AbstractStateSpace)
    @unpack A, B, C, D = P

    if size(D) != (1, 1)
        error("This is not a SISO system")
    end

    # Put into observer canonical form so the first element is also the y value
    BB = reverse(vec(C))
    CC = reverse(vec(B))'
    DD = D[1, 1]

    return function sim!(dx, x, p, t; u = 0.0)
        dx .= A * x + BB * u
        return CC * x + DD * u
    end
end

## Model setup
# We'll make a Laplace variable `s`
s = tf("s")

# Here is our reference model to track
am = 3
bm = 3
ref_model = bm / (s + am)
ref_sim! = SISO_simulator(ref_model)

# and our plant model. The nominal plant structure is what is known to our adaptation law.
ap = 1
bp = 2
nominal_plant = bp / (s + ap)
nominal_sim! = SISO_simulator(nominal_plant)

# To test robustness to uncertainty, we'll also include unmodeled dynamics with an entirely
# different structure than our nominal plant model.
unmodeled_dynamics = 229 / (s^2 + 30s + 229)
truth_plant = nominal_plant * unmodeled_dynamics
truth_sim! = SISO_simulator(truth_plant)

# We'll make a first-order sensor as well so we can add noise to our measurement
τ = 0.005
sensor_plant = 1 / (τ * s + 1)
sensor_sim! = SISO_simulator(sensor_plant)

## Derivative functions
# Our control law assumes perfect knowledge of the parameters that are attached to the
# regressors (which are the reference input and the model output)
control(θ, w) = θ'w

# We'll use a simple gradient descent adaptation law
function adapt!(Dθ, θ, γ, t; e, w)
    Dθ .= -γ * e * w
    return nothing
end

# Our feedback loop takes in the reference model output `ym` and the input signal `r`,
# calculates the control signal `u`, feeds that into the plant model, calculates the reference
# tracking error `e`, and finally updates feeds the reference tracking error and it's corresponding
# regressor vector to the adaptation law.
function feedback_sys!(D, vars, p, t; ym, r, n)
    @unpack parameter_estimates, plant_model, sensor = vars
    γ = p.gamma
    regressor = [r, plant_model[1]]

    u = control(parameter_estimates, regressor)
    yp = p.plant_fun(D.plant_model, plant_model, (), t; u = u)
    ŷ = sensor_sim!(D.sensor, sensor, (), t; u = yp[1]) + n
    e = ŷ .- ym
    regressor[2] = ŷ
    adapt!(D.parameter_estimates, parameter_estimates, γ, t; e = e, w = regressor)
    return yp
end
# Now the full system takes in an input signal `r`, feeds it through the reference model,
# and feeds the output of the reference model `ym` and the input signal to `feedback_sys`.
function system!(D, vars, p, t; r = 0.0, n = 0.0)
    @unpack reference_model, feedback_loop = vars

    ym = ref_sim!(D.reference_model, reference_model, (), t; u = r)
    yp = feedback_sys!(D.feedback_loop, feedback_loop, p, t; ym = ym, r = r, n = n)
    return yp
end

## Simulation inputs
# Simulation time span
tspan = (0.0, 30.0)

# Input signal
input_signal = (x, p, t) -> sin(3t)

# Initial conditions
ref_ic = zeros(1)
nominal_ic = zeros(1)
truth_ic = zeros(3)
sensor_ic = zeros(1)
θ_est_ic = ComponentArray(θr = 0.0, θy = 0.0)

## Set up and run Simulation
function simulate(
        plant_fun, plant_ic;
        tspan = tspan,
        input_signal = input_signal,
        adapt_gain = 1.5,
        noise_param = nothing,
        deterministic_noise = 0.0
    )
    noise(D, vars, p, t) = (D.feedback_loop.sensor[1] = noise_param)

    # Truth control parameters
    θ_truth = (r = bm / bp, y = (ap - am) / bp)

    # Initial conditions
    ic = ComponentArray(
        reference_model = ref_ic,
        feedback_loop = (
            parameter_estimates = θ_est_ic,
            sensor = sensor_ic,
            plant_model = plant_ic,
        )
    )

    # Model parameters
    p = (
        gamma = adapt_gain,
        plant_fun = plant_fun,
    )

    sim_fun = apply_inputs(system!; r = input_signal, n = deterministic_noise)

    # We can also choose whether we want to include noise in our model by switching between an ODE
    # and an SDE problem.
    if noise_param === nothing
        prob = ODEProblem(sim_fun, ic, tspan, p, max_iters = 2000)
    else
        prob = SDEProblem(sim_fun, noise, ic, tspan, p, max_iters = 2000) # With noise
    end

    ## Solve!
    sol = solve(prob)

    ## Plot
    # Reference model tracking
    top = plot(
        sol,
        vars = Symbol.(["reference_model[1]", "feedback_loop.sensor[1]"]),
        # legend=:right,
        title = "Reference Model Tracking"
    )

    # Parameter estimate tracking
    bottom = plot(
        sol,
        vars = Symbol.(
            [
                "feedback_loop.parameter_estimates.θr", "feedback_loop.parameter_estimates.θy",
            ]
        )
    )
    plot!(
        bottom,
        [tspan...], [θ_truth.r θ_truth.y; θ_truth.r θ_truth.y],
        labels = ["θr truth" "θy truth"],
        legend = :right,
        title = "Parameter Estimate Tracking"
    )

    # Combine both plots
    return plot(top, bottom, layout = (2, 1), size = (800, 800))
end

simulate(
    truth_sim!, truth_ic; input_signal = 2.0,
    deterministic_noise = (x, p, t) -> 0.5sin(16.1t), noise_param = nothing
)

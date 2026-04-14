# Control of a sliding block

```julia
using ComponentArrays
using DifferentialEquations
using Interact: @manipulate
using Parameters: @unpack
using Plots
```

## Problem Setup

```julia
const g = 9.80665

maybe_apply(f::Function, x, p, t) = f(x, p, t)
maybe_apply(f, x, p, t) = f

# Applies functions of form f(x,p,t) to be applied and passed in as inputs
function simulator(func; kwargs...)
    simfun(dx, x, p, t) = func(
        dx, x, p, t; map(f->maybe_apply(f, x, p, t), (; kwargs...))...)
    simfun(x, p, t) = func(x, p, t; map(f->maybe_apply(f, x, p, t), (; kwargs...))...)
    return simfun
end

softsign(x) = tanh(1e3x)
```

## Component Functions

### A sliding block with two different friction models

```julia
# Sliding block with viscous friction
function viscous_block!(D, vars, p, t; u = 0.0)
    @unpack m, c, k = p
    @unpack v, x = vars

    D.x = v
    D.v = (-c*v + k*(u-x))/m
    return x
end

# Sliding block with coulomb friction
function coulomb_block!(D, vars, p, t; u = 0.0)
    @unpack m, μ, k = p
    @unpack v, x = vars

    D.x = v
    a = -μ*g*softsign(v) + k*(u-x)/m
    D.v = abs(a)<1e-3 && abs(v)<1e-3 ? -10v : a #deadzone to help the simulation
    return x
end
```

### PID feedback control

```julia
function PID_controller!(D, vars, p, t; err = 0.0, v = 0.0)
    @unpack kp, ki, kd = p
    @unpack x = vars

    D.x = ki*err
    return x + kp*err + kd*v
end

function feedback_sys!(D, components, p, t; ref = 0.0)
    @unpack ctrl, plant = components

    u = p.ctrl.fun(D.ctrl, ctrl, p.ctrl.params, t; err = ref-plant.x, v = -plant.v)
    return p.plant.fun(D.plant, plant, p.plant.params, t; u)
end

step_input(; time = 1.0, mag = 1.0) = (x, p, t) -> t>time ? mag : 0
sine_input(; mag = 1.0, period = 10.0) = (x, p, t) -> mag*sin(t*2π/period)

# Equivalent viscous damping coefficient taken from:
# https://engineering.purdue.edu/~deadams/ME563/lecture2010.pdf
visc_equiv(μ, N, ω, mag) = 4*μ*N/(π*ω*mag)
```

## Open-Loop Response

To see the open-loop response of the coulomb system, let's set the input to `5` and plot
the results.

```julia
const tspan = (0.0, 30.0)
const m = 50.0
const μ = 0.1
const k = 50.0

p = (; m, μ, k)
ic = ComponentArray(v = 0, x = 0)

ODEProblem(simulator(coulomb_block!, u = 5), ic, tspan, p) |> solve |> plot
```

![](../assets/simple_coulomb.png)

## Closed-Loop Response

For the closed-loop response, let's make an interactive GUI. Since we are using
`ComponentArray`s, we don't have to change anything about our plant model to incorporate
it in the overall system simulation.

```julia
p = (
    ctrl = (
        params = (kp = 13, ki = 12, kd = 5),
        fun = PID_controller!
    ),
    plant = (
        params = plant_p,
        fun = coulomb_block!
    )
)

ic = ComponentArray(ctrl = (; x = 0), plant = plant_ic)

sol = ODEProblem(simulator(feedback_sys!, ref = 10), ic, tspan, p) |> solve
plot(sol, vars = 3)
```

```julia
## Interactive GUI for switching out plant models and varying PID gains
@manipulate for kp in 0:0.01:15,
    ki in 0:0.01:15,
    kd in 0:0.01:15,
    damping in Dict(
        "Coulomb" => coulomb_block!,
        "Viscous" => viscous_block!
    ),
    reference in Dict(
        "Sine" => sine_input,
        "Step" => step_input
    ),
    magnitude in 0:0.01:10, # pop-pop!
    period in 1:0.01:30,
    plot_v in false

    # Inputs
    tspan = (0.0, 30.0)

    ctrl_fun = PID_controller!
    # plant_fun = coulomb_block!

    ref = if reference==sine_input
        reference(; period, mag = magnitude)
    else
        reference(mag = magnitude)
    end

    m = 50.0
    μ = 0.1
    ω = 2π/period
    c = 4*μ*m*g/(π*ω*magnitude) # Viscous equivalent damping
    k = 50.0

    plant_p = (; m, μ, c, k) # We'll just put everything for both models in here
    ctrl_p = (; kp, ki, kd)

    plant_ic = (v = 0, x = 0)
    ctrl_ic = (; x = 0)

    # Set up and solve
    sys_p = (
        ctrl = (params = ctrl_p, fun = ctrl_fun),
        plant = (params = plant_p, fun = damping)
    )
    sys_ic = ComponentArray(ctrl = ctrl_ic, plant = plant_ic)
    sys_fun = ODEFunction(simulator(feedback_sys!; ref), syms = [:u, :v, :x])
    sys_prob = ODEProblem(sys_fun, sys_ic, tspan, sys_p)

    sol = solve(sys_prob, Tsit5())

    # Plot
    t = tspan[1]:0.1:tspan[2]
    lims = magnitude*[-1, 1]
    plotvars = plot_v ? [3, 2] : [3]
    strip = plot(t, ref.(0, 0, t), ylim = 1.2lims, label = "r(t)")
    plot!(strip, sol, vars = plotvars)
    phase = plot(ref.(0, 0, t), map(x->x.plant.x, sol(t).u),
        xlim = lims,
        ylim = 1.2lims,
        legend = false,
        xlabel = "r(t)",
        ylabel = "x(t)"
    )
    plot(strip, phase, layout = (2, 1), size = (700, 800))
end
```

![](../assets/coulomb_control.png)

## This example comes from https://diffeqflux.sciml.ai/dev/Flux/

using ComponentArrays
using OrdinaryDiffEq
using Plots
using UnPack

using DiffEqFlux: sciml_train
using Flux: glorot_uniform, ADAM, σ, relu
using Optim: LBFGS

struct MeasurementNoise{T}
    sigma::T
end
Base.:(+)(array, mn::MeasurementNoise) = array .+ randn(size(array)...) .* mn.sigma

# Problem setup
u0 = Float32[2.0; 0.0]
datasize = 50
tspan = (0.0f0, 7.0f0)
tspan2 = (0.0f0, 25.0f0)

# Make truth data
function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    return du .= ((u .^ 3)'true_A)'
end

t = range(tspan[1], tspan[2], length = datasize)
# t = Float32.(vcat(range(0.0, 0.9, length=10), 10 .^ range(log10(tspan[1]+1), log10(tspan[2]), length=datasize-10)))
t = Float32.([0; 10 .^ range(log10(tspan[1] + 0.01), log10(tspan[2]), length = datasize - 1)])
prob = ODEProblem(trueODEfunc, u0, tspan2)
ode_sol = solve(prob, Tsit5())
ode_data = Array(ode_sol(t)) + MeasurementNoise(0.1)

# Function for creating neural layer components
neural_layer(in, out) = ComponentArray{Float32}(W = glorot_uniform(out, in), b = zeros(out))

# Dense neural layer function
dense(layer, activation = identity) = u -> activation.(layer.W * u + layer.b)

# Neural ODE function
dudt(u, p, t) = u .^ 3 |> dense(p.L1, σ) |> dense(p.L2)

prob = ODEProblem(dudt, u0, tspan2)

# Create optimization parameter vector
n_hidden = 50
layers = (L1 = neural_layer(2, n_hidden), L2 = neural_layer(n_hidden, 2))
θ = ComponentArray(u = u0, p = layers)

# Prediction and loss functions
predict_n_ode(θ) = Array(solve(prob, Tsit5(), u0 = θ.u, p = θ.p, saveat = t))
full_sol(θ) = solve(prob, Tsit5(), u0 = θ.u, p = θ.p)

function loss_n_ode(θ)
    pred = predict_n_ode(θ)

    loss = sum(abs2, ode_data .- pred) / datasize + 0.1 * (sum(abs, θ.p) / length(θ.p))
    return loss, pred
end
loss_n_ode(θ)

# anim = Animation()

# Callback function to observe training
cb = function (θ, loss, pred; doplot = false)
    display(loss)
    # plot current prediction against data
    train_sol = full_sol(θ)

    pl_1 = plot(train_sol, vars = 1)
    plot!(pl_1, ode_sol, vars = 1)
    scatter!(pl_1, t, pred[1, :])
    scatter!(pl_1, t, ode_data[1, :], legend = false)

    pl_2 = plot(train_sol, vars = 2)
    plot!(pl_2, ode_sol, vars = 2)
    scatter!(pl_2, t, pred[2, :])
    scatter!(pl_2, t, ode_data[2, :], legend = false)

    pl_3 = plot(train_sol, vars = (1, 2), label = "prediction")
    plot!(pl_3, ode_sol, vars = (1, 2), label = "truth")
    scatter!(pl_3, pred[1, :], pred[2, :], label = "predicted data")
    scatter!(pl_3, ode_data[1, :], ode_data[2, :], label = "measured data")
    plot!(
        pl_3, hcat(pred[1, :], ode_data[1, :])', hcat(pred[2, :], ode_data[2, :])',
        label = false, color = :lightgray, legend = :bottomright
    )

    display(
        plot(
            plot(pl_1, pl_2, layout = (2, 1), size = (400, 500)), pl_3, layout = (1, 2), size = (
                950, 500,
            )
        )
    )
    # frame(anim)
    return false
end

# Display the ODE with the initial parameter values.
cb(θ, loss_n_ode(θ)...)

data = Iterators.repeated((), 1000)

res1 = sciml_train(loss_n_ode, θ, ADAM(0.05); maxiters = 100, save_best = true, cb = cb)
cb(res1.minimizer, loss_n_ode(res1.minimizer)...; doplot = true)

res2 = sciml_train(loss_n_ode, res1.minimizer, LBFGS(), cb = cb)
cb(res2.minimizer, loss_n_ode(res2.minimizer)...; doplot = true)

# gif(anim, "DiffEqFlux.gif", fps=15)

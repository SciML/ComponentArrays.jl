include("shared/test_setup.jl")

# Numeric checks compare against LinearAlgebra on the parent arrays. Broadcast
# `α .* x .+ β .* y` can differ from BLAS axpby! by 1 ULP on i686.

y = ComponentArray(a = rand(4), b = rand(4))
x = ComponentArray(a = rand(4), b = rand(4))
ydata = copy(getdata(y))
ystorage = getdata(y)

result = axpy!(2, x, y)
@test result === y
@test getdata(result) === ystorage
@test getdata(y) == axpy!(2, getdata(x), ydata)

previous = copy(getdata(y))
result = axpy!(2, x, result)
@test result === y
@test getdata(result) === ystorage
@test getdata(y) == axpy!(2, getdata(x), previous)

x = ComponentArray(a = rand(4), c = rand(4))
@test_throws ArgumentError axpy!(2, x, y)

y = ComponentArray(a = rand(4), b = rand(4))
x = ComponentArray(a = rand(4), b = rand(4))
ydata = copy(getdata(y))
ystorage = getdata(y)

result = axpby!(2, x, 3, y)
@test result === y
@test getdata(result) === ystorage
@test getdata(y) == axpby!(2, getdata(x), 3, ydata)

previous = copy(getdata(y))
result = axpby!(1, x, 1, result)
@test result === y
@test getdata(result) === ystorage
@test getdata(y) == axpby!(1, getdata(x), 1, previous)

x = ComponentArray(a = rand(4), c = rand(4))
@test_throws ArgumentError axpby!(2, x, 3, y)

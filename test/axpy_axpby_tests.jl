include("shared/test_setup.jl")

y = ComponentArray(a = rand(4), b = rand(4))
x = ComponentArray(a = rand(4), b = rand(4))
ydata = copy(getdata(y))
ystorage = getdata(y)

result = axpy!(2, x, y)
@test result === y
@test getdata(result) === ystorage
@test getdata(y) == 2 .* getdata(x) .+ ydata

previous = copy(getdata(y))
result = axpy!(2, x, result)
@test result === y
@test getdata(result) === ystorage
@test getdata(y) == 2 .* getdata(x) .+ previous

x = ComponentArray(a = rand(4), c = rand(4))
@test_throws ArgumentError axpy!(2, x, y)

y = ComponentArray(a = rand(4), b = rand(4))
x = ComponentArray(a = rand(4), b = rand(4))
ydata = copy(getdata(y))
ystorage = getdata(y)

result = axpby!(2, x, 3, y)
@test result === y
@test getdata(result) === ystorage
@test getdata(y) == 2 .* getdata(x) .+ 3 .* ydata

previous = copy(getdata(y))
result = axpby!(1, x, 1, result)
@test result === y
@test getdata(result) === ystorage
@test getdata(y) == getdata(x) .+ previous

x = ComponentArray(a = rand(4), c = rand(4))
@test_throws ArgumentError axpby!(2, x, 3, y)

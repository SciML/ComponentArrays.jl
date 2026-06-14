include("shared/test_setup.jl")

y = ComponentArray(a = rand(4), b = rand(4))
x = ComponentArray(a = rand(4), b = rand(4))
ydata = copy(getdata(y))

axpy!(2, x, y)
@test getdata(y) == 2 .* getdata(x) .+ ydata

x = ComponentArray(a = rand(4), c = rand(4))
@test_throws ArgumentError axpy!(2, x, y)

y = ComponentArray(a = rand(4), b = rand(4))
x = ComponentArray(a = rand(4), b = rand(4))
ydata = copy(getdata(y))

axpby!(2, x, 3, y)
@test getdata(y) == 2 .* getdata(x) .+ 3 .* ydata

x = ComponentArray(a = rand(4), c = rand(4))
@test_throws ArgumentError axpby!(2, x, 3, y)

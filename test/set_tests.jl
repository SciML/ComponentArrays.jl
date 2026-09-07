include("shared/test_setup.jl")

temp = deepcopy(ca2)
tempmat = deepcopy(cmat2)

temp.c.a .= 1000

view(view(tempmat, :b, :b)[1, 1], :a, :a)[:a, :a] = 100000
@view(tempmat[:b, :a])[2].b = 1000

@test temp.c.a.a == 1000

@test tempmat["b", "b"][1, 1]["a", :a][:a, :a] == 100000
@test tempmat[:b, :a][2].b == 1000

temp_b = deepcopy(temp.b)
temp.b .= temp.b .* 100
@test temp.b[1] == temp_b[1] .* 100

temp2 = deepcopy(ca)
temp3 = deepcopy(ca_MVector)
@test (temp2 .= ca .* 1) isa ComponentArray
@test (temp2 .= temp2 .* a .+ 1) isa typeof(temp2)
@test (temp2 .= ca .* ca_SVector) isa typeof(temp2)
@test (temp3 .= ca .* ca_SVector) isa typeof(temp3)

temp2.b = ca.b .+ 1
@test temp2.b == ca.b .+ 1

setproperty!(temp2, :a, 20)
@test temp2.a == 20

setproperty!(temp2, Val(:b), zeros(2))
@test temp2.b == zeros(2)

tempmat .= 0
@test tempmat[:b, :a][2].b == 0

temp = deepcopy(cmat)
@test all((temp[:c, :c][:a, :a] .= 0) .== 0)

A = ComponentArray(zeros(Int, 4, 4), Axis(x = r2v(1:4)), Axis(x = r2v(1:4)))
A[1, :] .= 1
@test A[1, :] == ComponentVector(x = ones(Int, 4))

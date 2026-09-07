include("shared/test_setup.jl")

@test ca == ComponentArray(
    a = 100, b = [4, 1.3], c = (
        a = (a = 1, b = [1.0, 4.4]), b = [0.4, 2, 1, 45],
    )
)
@test ca_Float32 == ComponentArray(Float32.(a), ax)
@test eltype(ComponentArray{ForwardDiff.Dual}(nt)) == ForwardDiff.Dual
@test ca_composed.b isa ComponentArray
@test ca_composed.b == ca
@test getdata(ca_MVector) isa MArray
@test typeof(ComponentArray(undef, (ax,))) == typeof(ca)
@test typeof(ComponentArray(undef, (ax, ax))) == typeof(cmat)
@test typeof(ComponentArray{Float32}(undef, (ax,))) == typeof(ca_Float32)
@test typeof(ComponentArray{MVector{10, Float64}}(undef, (ax,))) == typeof(ca_MVector)

# Entry from Dict
dict1 = Dict(:a => rand(5), :b => rand(5, 5))
dict2 = Dict(:a => 3, :b => dict1)
@test ComponentArray(dict1) isa ComponentArray
@test ComponentArray(dict2).b isa ComponentArray

@test ca == ComponentVector(
    a = 100, b = [4, 1.3], c = (
        a = (a = 1, b = [1.0, 4.4]), b = [0.4, 2, 1, 45],
    )
)
@test cmat == ComponentMatrix(a .* a', ax, ax)
@test_throws DimensionMismatch ComponentVector(sq_mat, ax)
@test_throws DimensionMismatch ComponentMatrix(rand(11, 11, 11), ax, ax)
@test_throws ErrorException ComponentArray(v = [(a = 1, b = 2), (a = 3, c = 4)])

# Axis construction from symbols
@test Axis([:a, :b, :c]) == Axis(a = 1, b = 2, c = 3)
@test Axis((:a, :b, :c)) == Axis(a = 1, b = 2, c = 3)
@test Axis(:a, :b, :c) == Axis(a = 1, b = 2, c = 3)
@test_throws ErrorException Axis(:a, :a)

# Issue #24
@test ComponentVector(a = 1, b = 2.0f0) == ComponentVector{Float32}(a = 1.0, b = 2.0)
@test ComponentVector(a = 1, b = 2 + im) ==
    ComponentVector{Complex{Int64}}(a = 1 + 0im, b = 2 + 1im)

# Issue #23
sz = size(ca)
temp = ComponentArray(ca; d = 100)
temp2 = ComponentVector(temp; d = 4)
temp3 = ComponentArray(temp2; e = (a = 20, b = [2 4; 1 4]))
@test sz == size(ca)
@test temp.d == 100
@test temp2.d == 4
@test !haskey(ca, :d)
@test all(temp3.e.b .== [2 4; 1 4])

# Issue #18
temp_miss = ComponentArray(a = missing, b = [2, 1, 4, 5], c = [1, 2, 3])
@test eltype(temp_miss) == Union{Int64, Missing}
@test temp_miss.a === missing
temp_noth = ComponentArray(a = nothing, b = [2, 1, 4, 5], c = [1, 2, 3])
@test eltype(temp_noth) == Union{Int64, Nothing}
@test temp_noth.a === nothing

# Issue #61
@test ComponentArray(x = 1) isa ComponentArray{Int}

# Issue #81
@test ComponentArray() isa ComponentArray
@test ComponentVector() isa ComponentVector
@test ComponentMatrix() isa ComponentMatrix
@test ComponentArray{Float32}() isa ComponentArray{Float32}
@test ComponentVector{Float32}() isa ComponentVector{Float32}
@test ComponentMatrix{Float32}() isa ComponentMatrix{Float32}

# Issue #116
# Part 2: Arrays of arrays
@test_throws Exception ComponentVector(a = [[3], [4, 5]], b = 1)

x = ComponentVector(a = [[3, 3], [4, 5]], b = 1)
@test x.a[1] == [3, 3]
@test x.b == 1

# empty components
for T in [Int64, Int32, Float64, Float32, ComplexF64, ComplexF32]
    @test ComponentArray(a = T[]) == ComponentVector{T}(a = T[])
    @test ComponentArray(a = T[], b = T[]) == ComponentVector{T}(a = T[], b = T[])
    @test ComponentArray(a = T[], b = (;)) == ComponentVector{T}(a = T[], b = T[])
    @test ComponentArray(a = Any[one(Int32)], b = T[]) ==
        ComponentVector{T}(a = [one(T)], b = T[])
end
@test ComponentArray(NamedTuple()) == ComponentVector{Any}()
@test ComponentArray(a = []).a == []

# Make sure type promotion works correctly with StaticArrays of NamedTuples
@test ComponentVector(a = SA[(a = 2, b = true)], b = false) isa ComponentVector{Int}

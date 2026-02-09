using ComponentArrays
using JLArrays, LinearAlgebra
using Test

JLArrays.allowscalar(false)

jla = jl(collect(1:4))
jlca = ComponentArray(jla, Axis(a = 1:2, b = 3:4))

@testset "Broadcasting" begin
    @test identity.(jlca + jla) ./ 2 == jlca

    @test getdata(map(identity, jlca)) isa JLArray
    @test all(==(0), map(-, jlca, jla))
    @test all(map(-, jlca, jlca) .== 0)
    @test all(==(0), map(-, jla, jlca)) broken = (pkgversion(JLArrays.GPUArrays) ≥ v"11")

    @test any(==(1), jlca)
    @test count(>(2), jlca) == 2

    # Make sure mapreducing multiple arrays works
    @test mapreduce(==, +, jlca, jla) == 4
    @test mapreduce(abs2, +, jlca) == 30

    @test all(map(sin, jlca) .== sin.(jlca) .== sin.(jla) .≈ sin.(1:4))

    # Issue #179
    @test similar(jlca, 5) isa typeof(jla)
end

@testset "adapt" begin
    x = [1 2; 3 4]
    jlx = JLArrays.Adapt.adapt(typeof(jlca), x)
    @test jlx isa JLArray
end

@testset "Linear Algebra" begin
    @testset "fill!" begin
        jlca2 = deepcopy(jlca)
        jlca2 = fill!(jlca2, 2)
        @test jlca2 == ComponentArray(jl([2, 2, 2, 2]), Axis(a = 1:2, b = 3:4))
    end

    @testset "norm" begin
        @test norm(jlca, 2) == norm(jla, 2)
        @test norm(jlca, Inf) == norm(jla, Inf)
    end

    @testset "rmul!" begin
        jlca3 = deepcopy(jlca)
        @test rmul!(jlca3, 2) == ComponentArray(jla .* 2, Axis(a = 1:2, b = 3:4))
    end
    @testset "mul!" begin
        A = jlca .* jlca'
        @test_nowarn mul!(deepcopy(A), A, A, 1, 2)
        @test_nowarn mul!(deepcopy(A), A', A', 1, 2)
        @test_nowarn mul!(deepcopy(A), A', A, 1, 2)
        @test_nowarn mul!(deepcopy(A), A, A', 1, 2)
        @test_nowarn mul!(deepcopy(A), A, getdata(A'), 1, 2)
        @test_nowarn mul!(deepcopy(A), getdata(A'), A, 1, 2)
        @test_nowarn mul!(deepcopy(A), getdata(A'), getdata(A'), 1, 2)
        @test_nowarn mul!(deepcopy(A), transpose(A), A, 1, 2)
        @test_nowarn mul!(deepcopy(A), A, transpose(A), 1, 2)
        @test_nowarn mul!(deepcopy(A), transpose(A), transpose(A), 1, 2)
        @test_nowarn mul!(deepcopy(A), transpose(getdata(A)), A, 1, 2)
        @test_nowarn mul!(deepcopy(A), A, transpose(getdata(A)), 1, 2)
        @test_nowarn mul!(deepcopy(A), transpose(getdata(A)), transpose(getdata(A)), 1, 2)
        @test_nowarn mul!(deepcopy(A), transpose(A), A', 1, 2)
        @test_nowarn mul!(deepcopy(A), A', transpose(A), 1, 2)
    end
end

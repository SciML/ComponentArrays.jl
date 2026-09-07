include("shared/test_setup.jl")

let
    ca = ComponentArray(a = 1, b = 2, c = [3, 4], d = (a = [5, 6, 7], b = 8))
    cmat = ca * ca'

    cidx = reshape((1:(2 * 3)) .+ 2, 2, 3)
    ca2 = ComponentArray(a = 1, b = 2, c = cidx, d = (a = [9, 10, 11], b = 12))

    @testset "ComponentIndex" begin
        ax = getaxes(ca)[1]
        @test ax[:a] == ax[1] ==
            ComponentArrays.ComponentIndex(1, ComponentArrays.NullAxis())
        @test ax[:c] == ax[3:4] ==
            ComponentArrays.ComponentIndex(3:4, ShapedAxis(size(3:4)))
        @test ax[:d] == ComponentArrays.ComponentIndex(5:8, Axis(a = r2v(1:3), b = 4))
        @test ax[(:a, :c)] == ax[[:a, :c]] ==
            ComponentArrays.ComponentIndex([1, 3, 4], Axis(a = 1, c = r2v(2:3)))
        ax2 = getaxes(ca2)[1]
        @test ax2[(:a, :c)] == ax2[[:a, :c]] ==
            ComponentArrays.ComponentIndex(
                [1, 3:8...], Axis(a = 1, c = ViewAxis(2:7, ShapedAxis((2, 3))))
            )

        @test length(ComponentArrays.ComponentIndex(1, ComponentArrays.NullAxis())) == 1
        @test length(ComponentArrays.ComponentIndex(3:4, ShapedAxis(size(3:4)))) == 2
        @test length(ComponentArrays.ComponentIndex(5:8, Axis(a = r2v(1:3), b = 4))) ==
            4
        @test length(ComponentArrays.ComponentIndex([1, 3, 4], Axis(a = 1, c = r2v(2:3)))) ==
            3
        @test length(
            ComponentArrays.ComponentIndex(
                [1, 3:8...], Axis(a = 1, c = ViewAxis(2:7, ShapedAxis((2, 3))))
            )
        ) == 7
    end

    @testset "KeepIndex" begin
        @test ca[KeepIndex(:a)] == ca[KeepIndex(1)] == ComponentArray(a = 1)
        @test ca[KeepIndex(:b)] == ca[KeepIndex(2)] == ComponentArray(b = 2)
        @test ca[KeepIndex(:c)] == ca[KeepIndex(3:4)] == ComponentArray(c = [3, 4])
        @test ca[KeepIndex(:d)] == ca[KeepIndex(5:8)] ==
            ComponentArray(d = (a = [5, 6, 7], b = 8))

        @test ca[KeepIndex(1:2)] == ComponentArray(a = 1, b = 2)
        @test ca[KeepIndex(1:3)] == ComponentArray([1, 2, 3], Axis(a = 1, b = 2)) # Drops c axis
        @test ca[KeepIndex(2:5)] ==
            ComponentArray([2, 3, 4, 5], Axis(b = 1, c = r2v(2:3)))
        @test ca[KeepIndex(3:end)] ==
            ComponentArray(c = [3, 4], d = (a = [5, 6, 7], b = 8))

        @test ca[KeepIndex(:)] == ca

        @test cmat[KeepIndex(:a), KeepIndex(:b)] ==
            ComponentArray(fill(2, 1, 1), Axis(a = 1), Axis(b = 1))
        @test cmat[KeepIndex(:), KeepIndex(:c)] ==
            ComponentArray((1:8) * (3:4)', getaxes(ca)[1], Axis(c = r2v(1:2)))
        @test cmat[KeepIndex(2:5), 1:2] ==
            ComponentArray((2:5) * (1:2)', Axis(b = 1, c = r2v(2:3)), ShapedAxis(size(1:2)))
        @test cmat[KeepIndex(2), KeepIndex(3)] ==
            ComponentArray(fill(2 * 3, 1, 1), Axis(b = 1), FlatAxis())
        @test cmat[KeepIndex(2), 3] == ComponentArray(b = 2 * 3)
    end
end

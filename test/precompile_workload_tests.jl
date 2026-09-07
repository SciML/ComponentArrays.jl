using ComponentArrays
using Test

@testset "Precompile workload" begin
    component = ComponentArray(position = [1.0, 2.0], velocity = [3.0, 4.0])

    @test component isa ComponentVector
    @test component.position == [1.0, 2.0]
    @test component[:velocity] == [3.0, 4.0]
    @test component[1:2] == [1.0, 2.0]
    @test getdata(component) == [1.0, 2.0, 3.0, 4.0]
    expected_axis = Axis(
        position = ViewAxis(1:2, ShapedAxis((2,))),
        velocity = ViewAxis(3:4, ShapedAxis((2,))),
    )
    @test getaxes(component) == (expected_axis,)
    @test axes(component) == axes(getdata(component))
end

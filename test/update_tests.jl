using ComponentArrays
using Test

@testset "ComponentArray Update" begin
    # Test 1: Basic update
    default = ComponentArray(sig = (mu = 1.0, sigma = 2.0), bg = 3.0)
    update = ComponentArray(sig = (mu = 1.1,), bg = 3.3)

    result = update_component_array(default, update)

    @test result.sig.mu ≈ 1.1
    @test result.sig.sigma ≈ 2.0
    @test result.bg ≈ 3.3

    # Test 2: Structure preservation
    @test typeof(result) == typeof(default)
    @test getfield(result, :axes) == getfield(default, :axes)

    # Test 3: Deep nested update
    deep_default = ComponentArray(
        outer = (
            inner = (x = 1.0, y = 2.0),
            other = 3.0,
        ),
        top = 4.0,
    )

    deep_update = ComponentArray(
        outer = (
            inner = (x = 10.0,),
        ),
    )

    deep_result = update_component_array(deep_default, deep_update)

    @test deep_result.outer.inner.x ≈ 10.0
    @test deep_result.outer.inner.y ≈ 2.0
    @test deep_result.outer.other ≈ 3.0
    @test deep_result.top ≈ 4.0

    # Test 4: Non-existing field update
    nonexist_update = ComponentArray(
        sig = (mu = 1.1,),
        nonexistent = 42.0,  # This field doesn't exist in default
    )
    result_nonexist = update_component_array(default, nonexist_update)

    # Should preserve structure and only update existing fields
    @test result_nonexist.sig.mu ≈ 1.1
    @test result_nonexist.sig.sigma ≈ 2.0
    @test !hasproperty(result_nonexist, :nonexistent)
    @test typeof(result_nonexist) == typeof(default)
    @test getfield(result_nonexist, :axes) == getfield(default, :axes)
end

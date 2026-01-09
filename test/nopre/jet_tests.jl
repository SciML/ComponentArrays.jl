using ComponentArrays
using JET
using Test

@testset "JET Static Analysis" begin
    # Create test ComponentArrays for analysis
    ca = ComponentArray(a = 1.0, b = [2.0, 3.0], c = (x = 4.0, y = 5.0))
    ca_simple = ComponentArray(a = 1.0, b = 2.0)
    cmat = ca .* ca'

    @testset "Core operations type stability" begin
        # Test getindex with integer - should be type stable
        rep = @report_opt target_modules = (ComponentArrays,) ca[1]
        @test length(JET.get_reports(rep)) == 0

        # Test similar - should be type stable
        rep = @report_opt target_modules = (ComponentArrays,) similar(ca)
        @test length(JET.get_reports(rep)) == 0

        # Test copy - should be type stable
        rep = @report_opt target_modules = (ComponentArrays,) copy(ca)
        @test length(JET.get_reports(rep)) == 0

        # Test getdata - should be type stable
        rep = @report_opt target_modules = (ComponentArrays,) getdata(ca)
        @test length(JET.get_reports(rep)) == 0

        # Test getaxes - should be type stable
        rep = @report_opt target_modules = (ComponentArrays,) getaxes(ca)
        @test length(JET.get_reports(rep)) == 0

        # Test broadcast - should be type stable
        rep = @report_opt target_modules = (ComponentArrays,) broadcast(+, ca, ca)
        @test length(JET.get_reports(rep)) == 0

        # Test vcat - should be type stable
        rep = @report_opt target_modules = (ComponentArrays,) vcat(ca_simple, ca_simple)
        @test length(JET.get_reports(rep)) == 0

        # Test hcat - should be type stable
        rep = @report_opt target_modules = (ComponentArrays,) hcat(ca_simple, ca_simple)
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "No runtime errors in basic usage" begin
        # Check that basic operations don't have potential runtime errors
        rep = @report_call ca[1]
        @test length(JET.get_reports(rep)) == 0

        rep = @report_call similar(ca)
        @test length(JET.get_reports(rep)) == 0

        rep = @report_call copy(ca)
        @test length(JET.get_reports(rep)) == 0

        rep = @report_call getdata(ca)
        @test length(JET.get_reports(rep)) == 0

        rep = @report_call getaxes(ca)
        @test length(JET.get_reports(rep)) == 0
    end
end

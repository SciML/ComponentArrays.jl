include("shared/test_setup.jl")

@test_deprecated ComponentArrays.getval.(fastindices(:a, :b, :c)) == (:a, :b, :c)
@test_deprecated fastindices(:a, Val(:b)) == (Val(:a), Val(:b))

@test collect(ComponentArrays.partition(collect(1:12), 3)) ==
    [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
@test size(collect(ComponentArrays.partition(zeros(2, 2, 2), 1, 2, 2))[2, 1, 1]) ==
    (1, 2, 2)

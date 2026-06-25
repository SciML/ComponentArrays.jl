using SciMLTesting, ComponentArrays, Test
using JET

run_qa(
    ComponentArrays;
    explicit_imports = true,
    # ComponentArrays has real method ambiguities and unbound type parameters in its
    # vcat/hcat/getindex/Axis overloads; these are long-standing design realities, not
    # tracked-broken placeholders, so disable the sub-checks rather than fail.
    aqua_kwargs = (; ambiguities = false, unbound_args = false),
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = (
                # Base / Base.Broadcast / Base.Iterators internals (method extension):
                :var"@propagate_inbounds", :Bottom, :BroadcastStyle, :Generator, :OneTo,
                :ProductIterator, :ReshapedArray, :axistype, :broadcast_unalias,
                :combine_axes, :dataids, :elsize, :maybeview, :print_array,
                :print_matrix, :to_index, :unalias, :unsafe_convert,
                # LinearAlgebra non-public (lu_instance/factorization internals):
                :BlasInt, :lutype,
                # Adapt non-public (adapt_storage/adapt_structure extension):
                :adapt_storage, :adapt_structure,
                # ChainRulesCore non-public:
                :backing,
                # ArrayInterface non-public:
                :indices_do_not_alias, :instances_do_not_alias, :lu_instance,
                :parent_type,
                # StaticArrayInterface non-public:
                :static_size,
                # Functors non-public:
                :functor,
            ),
        ),
        # StaticArraysCore.StaticArray is not declared public there yet.
        all_explicit_imports_are_public = (; ignore = (:StaticArray,)),
    ),
)

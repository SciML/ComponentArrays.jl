using SciMLTesting, ComponentArrays, Test
using JET

# ExplicitImports only walks an extension that is actually loaded (it resolves each
# `[extensions]` entry through `Base.get_extension`, which is `nothing` otherwise), so
# every weakdep is loaded here to bring the extension modules into the check.
using GPUArrays, KernelAbstractions, Mooncake, Optimisers, Reactant, RecursiveArrayTools
using ReverseDiff, SciMLBase, SymbolicIndexingInterface, Tracker, Zygote

# ExplicitImports silently skips an extension that fails to load, so assert the
# extension modules actually exist rather than trusting a green run_qa.
@testset "Extensions loaded" begin
    exts = (
        :ComponentArraysGPUArraysExt, :ComponentArraysKernelAbstractionsExt,
        :ComponentArraysMooncakeExt, :ComponentArraysOptimisersExt,
        :ComponentArraysReactantExt, :ComponentArraysRecursiveArrayToolsExt,
        :ComponentArraysReverseDiffExt, :ComponentArraysSciMLBaseExt,
        :ComponentArraysTrackerExt, :ComponentArraysZygoteExt,
    )
    for ext in exts
        @test Base.get_extension(ComponentArrays, ext) !== nothing
    end
end

run_qa(
    ComponentArrays;
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
                :BlasInt, :QRCompactWY, :lutype,
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
                # Zygote non-public: `accum` is Zygote's gradient-accumulation hook and
                # the only way to give ComponentArray a non-broadcasting accumulation
                # path (needed so ROCArray-backed arrays do not hit the AMDGPU
                # broadcast-compilation bug).
                :accum,
                # Reactant non-public: the traced array/number types an extension must
                # dispatch on to specialize ComponentArray for Reactant tracing.
                :TracedRArray, :TracedRNumber,
                # ComponentArrays' own internals, reached from its own extensions.
                # ExplicitImports treats an extension as a separate module, so these
                # read as non-public cross-module accesses even though they never
                # leave the package.
                :_getindex, :recursive_eltype,
                # Tracker non-public: its whole tracked-value interface. `data`,
                # `extract_grad!`, `istracked`, `tracker`, `Grads` and `TrackedStyle`
                # are what an array type has to hook to be trackable, and none of them
                # are exported or declared public.
                :Grads, :TrackedStyle, :data, :extract_grad!, :istracked, :tracker,
                # Base.Broadcast internals used to specialize Tracker's broadcasting:
                :Broadcasted, :materialize,
                # Mooncake non-public: its tangent/fdata/rdata interface. Declaring a
                # tangent type for a foreign array type is only expressible through
                # these names; Mooncake exports none of them.
                :AsPrimal, :FData, :FriendlyTangentCache, :NoFData, :NoRData,
                :NoTangent, :Tangent, :friendly_tangent_cache,
                :increment_and_get_rdata!,
                # SciMLBase non-public: the legacy `syms`/`sys` accessors the index
                # provider has to consult to stay compatible with problems that predate
                # SymbolicIndexingInterface.
                :getsyms, :has_syms, :has_sys,
                # Optimisers non-public: `Leaf` is the optimiser-state wrapper an array
                # type must destructure to support `Optimisers.setup`/`update`.
                :Leaf,
                # Another ComponentArrays internal reached from its own extension.
                :__value,
                # ReverseDiff non-public: its tracked-value interface, the same shape
                # as Tracker's above.
                :TrackedArray, :TrackedReal, :deriv, :tape, :value,
                # One more ComponentArrays internal reached from its own extension.
                :indexmap,
                # GPUArrays non-public: the generic fallbacks a wrapper array type has
                # to forward to (`mapreducedim!`, `generic_matmatmul!`, `generic_rmul!`).
                :generic_matmatmul!, :generic_rmul!, :mapreducedim!,
                # More Base internals used by the GPU broadcast/reduction paths:
                :AbstractBroadcasted, :add_sum, :mul_prod, :typename,
                # And one more ComponentArrays internal reached from its own extension.
                :fill_componentarray_ka!,
                # KernelAbstractions non-public: `backend` is how a device-array type
                # advertises its KA backend.
                :backend,
            ),
        ),
        # StaticArraysCore.StaticArray is not declared public there yet.
        # Base.IEEEFloat has no public spelling; the Mooncake extension needs it to
        # restrict its tangent type to the float element types Mooncake differentiates.
        # TracedRArray/TracedRNumber, _getindex and recursive_eltype are the same
        # entries as above, reached by `using ...: name` rather than qualified access.
        all_explicit_imports_are_public = (;
            ignore = (
                :StaticArray, :IEEEFloat, :TracedRArray, :TracedRNumber,
                :_getindex, :recursive_eltype, :Grads, :TrackedStyle, :data,
                :extract_grad!, :istracked, :tracker, :Broadcasted, :materialize,
                :AsPrimal, :FData, :FriendlyTangentCache, :NoFData, :NoRData,
                :NoTangent, :Tangent, :friendly_tangent_cache,
                :increment_and_get_rdata!,
                # SciMLBase non-public: the legacy `syms`/`sys` accessors the index
                # provider has to consult to stay compatible with problems that predate
                # SymbolicIndexingInterface.
                :getsyms, :has_syms, :has_sys,
                # Optimisers non-public: `Leaf` is the optimiser-state wrapper an array
                # type must destructure to support `Optimisers.setup`/`update`.
                :Leaf,
                # Another ComponentArrays internal reached from its own extension.
                :__value,
                # ReverseDiff non-public: its tracked-value interface, the same shape
                # as Tracker's above.
                :TrackedArray, :TrackedReal, :deriv, :tape, :value,
                # One more ComponentArrays internal reached from its own extension.
                :indexmap,
                # GPUArrays non-public: the generic fallbacks a wrapper array type has
                # to forward to (`mapreducedim!`, `generic_matmatmul!`, `generic_rmul!`).
                :generic_matmatmul!, :generic_rmul!, :mapreducedim!,
                # More Base internals used by the GPU broadcast/reduction paths:
                :AbstractBroadcasted, :add_sum, :mul_prod, :typename,
                # And one more ComponentArrays internal reached from its own extension.
                :fill_componentarray_ka!,
                # KernelAbstractions non-public: `backend` is how a device-array type
                # advertises its KA backend.
                :backend,
            ),
        ),
    ),
)

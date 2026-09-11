using ComponentArrays, BenchmarkTools
using LinearAlgebra, StableRNGs

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

# =============================================================================
# ComponentVector
# =============================================================================

SUITE["componentvector"] = BenchmarkGroup()

a_data = rand(rng, 100)
b_c = rand(rng, 50)
b_d = rand(rng, 10, 10)
cv = ComponentArray(a = a_data, b = (c = b_c, d = b_d))
cv2 = ComponentArray(a = rand(rng, 100), b = (c = rand(rng, 50), d = rand(rng, 10, 10)))

SUITE["componentvector"]["construct"] = @benchmarkable ComponentArray(
    a = $a_data, b = (c = $b_c, d = $b_d)
)
SUITE["componentvector"]["getproperty"] = @benchmarkable $cv.b.c
SUITE["componentvector"]["getindex_symbol"] = @benchmarkable $cv[Val(:b)]
SUITE["componentvector"]["broadcast_add"] = @benchmarkable $cv .+ $cv2
SUITE["componentvector"]["norm"] = @benchmarkable norm($cv)
SUITE["componentvector"]["map"] = @benchmarkable map(sin, $cv)

# =============================================================================
# ComponentMatrix
# =============================================================================

SUITE["componentmatrix"] = BenchmarkGroup()

cm_data = rand(rng, 20, 20)
cm_ax = Axis(x = 1:10, y = 11:20)
cm_ax2 = Axis(u = 1:10, v = 11:20)
cm = ComponentMatrix(cm_data, cm_ax, cm_ax2)
cv_20 = ComponentVector(x = rand(rng, 10), y = rand(rng, 10))

SUITE["componentmatrix"]["construct"] = @benchmarkable ComponentMatrix(
    $cm_data, $cm_ax, $cm_ax2
)
SUITE["componentmatrix"]["getindex_symbol"] = @benchmarkable $cm[Val(:x), Val(:u)]
SUITE["componentmatrix"]["matmul"] = @benchmarkable $cm * $cv_20

# =============================================================================
# Views
# =============================================================================

SUITE["views"] = BenchmarkGroup()
SUITE["views"]["view_axis"] = @benchmarkable $cv[Val(:b)]
SUITE["views"]["getdata"] = @benchmarkable getdata($cv)

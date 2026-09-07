include("shared/test_setup.jl")

lab = labels(ca2)
@test lab == [
    "a",
    "b[1].a.a",
    "b[1].a.b",
    "b[1].b",
    "b[2].a.a",
    "b[2].a.b",
    "b[2].b",
    "c.a.a",
    "c.a.b[1]",
    "c.a.b[2]",
    "c.b[1,1]",
    "c.b[2,1]",
    "c.b[1,2]",
    "c.b[2,2]",
]
@test label2index(ca2, "c.b") == collect(11:14)

# Issue #74
lab2 = labels(
    ComponentArray(
        a = 1, aa = ones(2), ab = [(a = 1, aa = ones(2)), (a = 1, aa = ones(2))],
        ac = (a = 1, ab = ones(2, 2))
    )
)
@test label2index(lab2, "a") == [1]
@test label2index(lab2, "aa") == collect(2:3)
@test label2index(lab2, "ab") == collect(4:9)
@test label2index(lab2, "ab[1].aa") == collect(5:6)
@test label2index(lab2, "ac") == collect(10:14)
@test label2index(lab2, "ac.a") == [10]
@test label2index(lab2, "ac.ab") == collect(11:14)

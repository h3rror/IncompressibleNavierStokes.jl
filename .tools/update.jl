# Update all environments in the project

envs = (
    ".",
    "lib/NeuralClosure",
    "lib/NeuralClosure/test",
    "lib/PaperDC",
    "lib/SymmetryClosure",
    "test",
    "examples",
    "docs",
)

root = joinpath(@__DIR__, "..")

for e in envs
    e = joinpath(root, e)
    run(`julia --project=$e -e 'using Pkg; Pkg.update()'`)
end

using ReducedOrderModels
using Test

@testset  "Arithmetic" begin
    @test 1+1 == 2
end

@testset "Snapshots" begin
    include("mock-up setup.jl")
    snapshots = ReducedOrderModels.create_snapshots(;
        setup,
        ustart,
        Δt = 0.01,
        n_snap = 100,
    )
    @test length(snapshots) == 100
    @test typeof(snapshots) == Vector{Tuple{Matrix{Float64}, Matrix{Float64}}}

    @test snapshots[1] == vec2tuple(tuple2vec(snapshots[1],setup),setup)
end
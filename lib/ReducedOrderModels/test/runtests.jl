using ReducedOrderModels
using Test

using LinearAlgebra

ROM = ReducedOrderModels

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
    @test length(snapshots.u) == 100
    @test length(snapshots.t) == 100
    @test typeof(snapshots.u) == Vector{Tuple{Matrix{Float64}, Matrix{Float64}}}

    @test snapshots.u[1] == vec2tuple(tuple2vec(snapshots.u[1],setup),setup)

    r = 10
    ϕ,svd_,snapmat = ROM.compute_POD_basis(snapshots,r,setup)

    @test ϕ'*ϕ ≈ I(r)

    @test ROM.rom_project(ϕ,ϕ) ≈ I(r)
    a = rand(10)
    @test ROM.rom_project(ROM.rom_reconstruct(a,ϕ),ϕ) ≈ a
end


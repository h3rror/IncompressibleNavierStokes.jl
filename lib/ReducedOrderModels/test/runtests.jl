using ReducedOrderModels
ROM = ReducedOrderModels

using IncompressibleNavierStokes
INS = IncompressibleNavierStokes

using Test

using LinearAlgebra


@testset "Arithmetic" begin
    @test 1 + 1 == 2
end

@testset "Snapshots" begin
    include("mock-up setup.jl")
    snapshots =
        ReducedOrderModels.create_snapshots(; setup, ustart, Δt = 0.01, n_snap = 100)
    @test length(snapshots.u) == 100
    @test length(snapshots.t) == 100
    @test typeof(snapshots.u) == Vector{Tuple{Matrix{Float64},Matrix{Float64}}}

    @test snapshots.u[1] == vec2tuple(tuple2vec(snapshots.u[1], setup), setup)

    r = 10
    ϕ, svd_, snapmat = ROM.compute_POD_basis(snapshots, r, setup)

    @test ϕ' * ϕ ≈ I(r)

    @test ROM.rom_project(ϕ, ϕ) ≈ I(r)
    a = rand(10)
    @test ROM.rom_project(ROM.rom_reconstruct(a, ϕ), ϕ) ≈ a

    astart = ROM.rom_project(tuple2vec(ustart, setup), ϕ)
    nstep = 10
    Δt = 0.01
    a, t = ROM.rom_timestep_loop(ϕ; setup, nstep, astart, Δt)

    @test t ≈ nstep*Δt
    @test typeof(a) == typeof(astart)
    @test norm(INS.divergence(vec2tuple(ROM.rom_reconstruct(a,ϕ),setup),setup)) < 1e-8

    D_r, y_D = ROM.rom_diffusion_operator(ϕ, setup)
    @test D_r*astart + y_D ≈ ROM.rom_project(tuple2vec(INS.diffusion(ustart,setup),setup),ϕ)
    # test symmetry and negative semi-definiteness of D_r
    @test maximum(eigen(D_r).values) <= 0
    @test D_r ≈ D_r'
end

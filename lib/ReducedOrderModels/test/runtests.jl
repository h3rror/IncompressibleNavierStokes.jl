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
    a_rand = ROM.ROMstate(rand(r))
    u_rand_vec = ROM.rom_reconstruct(a_rand,ϕ)
    @test ROM.rom_project(u_rand_vec, ϕ) ≈ a_rand

    u_rand = vec2tuple(u_rand_vec,setup)

    astart = ROM.rom_project(tuple2vec(ustart, setup), ϕ)
    nstep = 10
    Δt = 0.01
    a, t = ROM.rom_timestep_loop(ϕ; setup, nstep, astart, Δt)

    @test t ≈ nstep * Δt
    @test typeof(a) == typeof(astart)
    @test norm(INS.divergence(vec2tuple(ROM.rom_reconstruct(a, ϕ), setup), setup)) < 1e-8

    D_r, y_D = ROM.rom_diffusion_operator(ϕ, setup)
    @test D_r * a_rand + y_D ≈
          ROM.rom_project(tuple2vec(INS.diffusion(u_rand, setup), setup), ϕ)
    # test symmetry and negative semi-definiteness of D_r
    @test maximum(eigen(D_r).values) <= 0
    @test D_r ≈ D_r'

    @test ROM.convection(ustart, ustart, setup) == INS.convection(ustart, setup)

    u1 = vec2tuple(ϕ[:, 1], setup)
    u2 = vec2tuple(ϕ[:, 2], setup)

    @test tuple2vec(u1, setup)' * tuple2vec(ROM.convection(u1, u2, setup), setup) < 1e-15 
    @test tuple2vec(u1, setup)' * tuple2vec(ROM.convection(u1, u1, setup), setup) < 1e-15

    C_r2,C_r1,y_C = ROM.rom_convection_operator(ϕ,setup)
    @test C_r2*kron(a_rand,a_rand) + C_r1*a_rand + y_C ≈
            ROM.rom_project(tuple2vec(INS.convection(u_rand, setup), setup), ϕ)
    # test block-skew-symmetry of C_r2
    C_r2_tensor = reshape(C_r2,r,r,r)
    @test norm(C_r2_tensor+permutedims(C_r2_tensor,[2 1 3])) <= sqrt(eps())

    ROM_setup = ROM.rom_operators(ϕ,setup)
    @test D_r == ROM_setup.D_r
    @test y_D == ROM_setup.y_D
    @test C_r2 == ROM_setup.C_r2
    @test C_r1 == ROM_setup.C_r1
    @test y_C == ROM_setup.y_C

    @test ROM.rom_diffusion(a_rand,ROM_setup) == D_r * a_rand + y_D
    @test ROM.rom_convection(a_rand,a_rand,ROM_setup) ==
                C_r2*kron(a_rand,a_rand) + C_r1*a_rand + y_C

    # test energy conservation of precomputed ROMconvection operator 
    # (for differing velocities)
    @test (a_rand'*C_r2*kron(astart,a_rand))[] < sqrt(eps())
    @test (a_rand'*C_r2*kron(a_rand,a_rand))[] < sqrt(eps())
    # ( ... )[] is used to cast 1x1 Matrix to scalar

    a_eff, t_eff = ROM.rom_timestep_loop_efficient(ROM_setup; setup, nstep, astart, Δt)
    @test t_eff ≈ t
    @test a_eff ≈ a
end

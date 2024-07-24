module ReducedOrderModels

using IncompressibleNavierStokes
# using IncompressibleNavierStokes: apply_bc_u!

# using LinearAlgebra

INS = IncompressibleNavierStokes

include("utils.jl")
include("INSupdates.jl")

"""
    create_snapshots(;
        setup,
        n_snap,
        ustart,
        Δt,
        method = RKMethods.RK44(; T = eltype(ustart[1])),
        psolver = default_psolver(setup),
    )

Create snapshots.
"""
function create_snapshots(;
    setup,
    n_snap,
    ustart,
    Δt,
    method = RKMethods.RK44(; T = eltype(ustart[1])),
    psolver = default_psolver(setup),
)
    tstart = 0.0
    tend = Δt * n_snap
    θ = nothing
    tempstart = nothing

    # Cache arrays for intermediate computations
    cache = IncompressibleNavierStokes.ode_method_cache(method, setup, ustart, tempstart)

    # Time stepper
    stepper = IncompressibleNavierStokes.create_stepper(
        method;
        setup,
        psolver,
        u = ustart,
        temp = tempstart,
        t = tstart,
    )

    u = [zero.(ustart) for i = 1:n_snap]
    t = zeros(n_snap)

    for i = 1:n_snap
        # Perform a single time step with the time integration method
        stepper = IncompressibleNavierStokes.timestep!(method, stepper, Δt; θ, cache)
        # snapshots[i] .= stepper.u
        u[i][1] .= stepper.u[1]
        u[i][2] .= stepper.u[2]
        t[i] = stepper.t
    end

    (; u, t)
end

"""
    rom_timestep_loop(ϕ;
    setup,
    nstep,
    astart,
    Δt = 0.01,
    tstart = 0,
    psolver = default_psolver(setup),
)

    Simple time-stepping method for ROMs;
    should later be replaced by multiple-dispatching existing time-stepping code
"""
function rom_timestep_loop(
    ϕ;
    setup,
    nstep,
    astart,
    Δt = 0.01,
    tstart = 0,
    psolver = default_psolver(setup),
)
    a = astart
    t = tstart
    for i = 1:nstep
        u_vec = rom_reconstruct(a, ϕ)
        u = vec2tuple(u_vec, setup)
        INS.apply_bc_u!(u, t, setup)
        F = INS.momentum(u, nothing, t, setup)
        INS.apply_bc_u!(F, t, setup; dudt = true)
        dudt = INS.project(F, setup; psolver)
        dudt_vec = tuple2vec(dudt, setup)
        dadt = rom_project(dudt_vec, ϕ)
        a = a + Δt * dadt
        t = t + Δt
    end

    a, t
end

"""
    rom_timestep_loop_efficient(ϕ;
    setup,
    nstep,
    astart,
    Δt = 0.01,
    tstart = 0,
    psolver = default_psolver(setup),
)

    Simple time-stepping method for ROMs with precomputed ROM operators;
    should later be replaced by multiple-dispatching existing time-stepping code
"""
function rom_timestep_loop_efficient(
    ROM_setup;
    setup,
    nstep,
    astart,
    Δt = 0.01,
    tstart = 0,
    psolver = default_psolver(setup),
)
    a = astart
    t = tstart
    for i = 1:nstep
        F = rom_momentum(a,a, ROM_setup)
        dadt = F
        a = a + Δt * dadt
        t = t + Δt
    end

    a, t
end

"""
    rom_momentum(a, b, ROM_setup)

    compute momentum equation RHS for convecting ROM coefficient vector a and 
    diffusing and convected ROM coefficient vector b
"""
function rom_momentum(a, b, ROM_setup)
    rom_diffusion(b, ROM_setup) + rom_convection(a, b, ROM_setup)
end

"""
    rom_operators(ϕ,setup)

    Collects precomputated ROM operators in ROM_setup to facilitate access
"""
function rom_operators(ϕ, setup)
    D_r, y_D = rom_diffusion_operator(ϕ, setup)
    C_r2, C_r1, y_C = rom_convection_operator(ϕ, setup)

    (; D_r, y_D, C_r2, C_r1, y_C)
end

"""
    rom_diffusion_operator(ϕ,setup)

    precompute ROM diffusion operator for ROM basis ϕ
"""
function rom_diffusion_operator(ϕ, setup)
    # @warn("assuming time-independent boundary conditions")
    projected_diffusion(u) =
        rom_project(tuple2vec(INS.diffusion(vec2tuple(u, setup), setup), setup), ϕ)

    y_D = projected_diffusion(0 * ϕ[:, 1])

    r = size(ϕ)[2]
    D_r = zeros(r, r)
    for i = 1:r
        D_r[:, i] = projected_diffusion(ϕ[:, i]) - y_D
    end

    D_r, y_D
end

"""
    rom_diffusion(a, ROM_setup)

    compute diffusion contribution to momentum equation for ROM coefficient vector a
"""
rom_diffusion(a, ROM_setup) = ROM_setup.D_r * a + ROM_setup.y_D

"""
    rom_convection_operator(ϕ, setup)

    precompute ROM diffusion operator for ROM basis ϕ

    note: for inhomogeneous boundary conditions and differing convecting and convected velocities,
    C_r1 should be split into 
    C_r1a[:,i] = projected_convection(u_i, u_0) - y_C
    C_r1b[:,i] = projected_convection(u_0, u_i) -  y_C
"""
function rom_convection_operator(ϕ, setup)
    # @warn("assuming time-independent boundary conditions")
    projected_convection(u, v) = rom_project(
        tuple2vec(convection(vec2tuple(u, setup), vec2tuple(v, setup), setup), setup),
        ϕ,
    )

    u_0 = 0 * ϕ[:, 1]
    y_C = projected_convection(u_0, u_0)

    r = size(ϕ)[2]
    C_r1 = zeros(r, r)
    C_r2 = zeros(r, r, r)
    for i = 1:r
        u_i = ϕ[:, i]
        C_r1[:, i] =
            projected_convection(u_i, u_0) + projected_convection(u_0, u_i) - 2 * y_C
        for j = 1:r
            u_j = ϕ[:, j]
            C_r2[:, i, j] = projected_convection(u_i, u_j) - projected_convection(u_i, u_0)
            -projected_convection(u_0, u_j) + y_C
        end
    end

    reshape(C_r2, r, r^2), C_r1, y_C
end

"""
    rom_convection(a, b, ROM_setup)

    compute convection contribution to momentum equation for convecting ROM coefficient vector a
    and convected ROM coefficient vector b,
    i.e. b'*C_r2*kron(a,b) = 0 for all a,b with a divergence-free
"""
function rom_convection(a, b, ROM_setup)
    # @warn("if a=/= b, assuming homogeneous boundary conditions")
    ROM_setup.C_r2 * kron(a, b) + ROM_setup.C_r1*a + ROM_setup.y_C
end

# # Option 1 (piracy)
# function INS.momentum(u::Vector, setup)
#     ...
# end

# # Option 2 -> preferred!!!
# struct ROMState
#     a::Vector
# end
# function INS.momentum(u::ROMState, setup)
#     ...
# end

# # Option 3: Functional
# timestep(; rhs = rom_momentum)

end # module ReducedOrderModels

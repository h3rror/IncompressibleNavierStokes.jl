module ReducedOrderModels

using IncompressibleNavierStokes
# using IncompressibleNavierStokes: apply_bc_u!

# using LinearAlgebra

INS = IncompressibleNavierStokes

include("utils.jl")

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
    should later be replaced by multiple-dsipatching existing time-stepping code
"""
function rom_timestep_loop(ϕ;
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
        u_vec = rom_reconstruct(a,ϕ)
        u = vec2tuple(u_vec,setup)
        INS.apply_bc_u!(u, t, setup)
        F = INS.momentum(u, nothing, t, setup)
        INS.apply_bc_u!(F, t, setup; dudt = true)
        dudt = INS.project(F, setup; psolver)
        dudt_vec = tuple2vec(dudt,setup)
        dadt = rom_project(dudt_vec,ϕ)
        a = a + Δt * dadt
        t = t + Δt
    end

    a,t
end


# # Option 1 (piracy)
# function INS.momentum(u::Vector, setup)
#     ...
# end

# # Option 2
# struct ROMState
#     a::Vector
# end
# function INS.momentum(u::ROMState, setup)
#     ...
# end

# # Option 3: Functional
# timestep(; rhs = rom_momentum)


end # module ReducedOrderModels

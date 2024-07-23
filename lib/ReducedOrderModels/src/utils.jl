using LinearAlgebra


"""
    tuple2vec(u,setup)

    Convert velocity field representation as tuple of matrices for the velocity components 
    into one concatenated vector
"""
function tuple2vec(u,setup)
    Iu = setup.grid.Iu
    sx, sy = u
    [sx[Iu[1]][:]; sy[Iu[2]][:]]
end


"""
    vec2tuple(v,setup)

    Convert velocity field representation as one concatenated vector into
    a tuple of matrices for the velocity componentes
"""
function vec2tuple(v,setup)
    (; Iu, N, Nu) = setup.grid
    nx = Nu[1][1] * Nu[1][2]
    ny = Nu[2][1] * Nu[2][2]
    vx = reshape(v[1:nx], Nu[1])
    vy = reshape(v[nx+1:nx+ny], Nu[2])
    u = (zeros(N), zeros(N))
    u[1][Iu[1]] .= vx
    u[2][Iu[2]] .= vy
    INS.apply_bc_u!(u, 0.0, setup)
    @warn("assuming time-independent boundary conditions")

    u
end

export tuple2vec
export vec2tuple

"""
    compute_POD_basis(snapshots,r,setup)

    Compute POD basis of dimension r from snapshot data snapshots
"""
function compute_POD_basis(snapshots,r,setup)
    vecshots = map(snapshots.u) do u
        tuple2vec(u,setup)
    end

    snapmat = stack(vecshots)

    svd_ = svd(snapmat)

    ϕ = svd_.U[:,1:r]

    ϕ,svd_,snapmat
end

"""
    rom_reconstruct(a,ϕ)

    Reconstruct FOM dimension velocity u from ROM coefficinet vector a and ROM basis ϕ
"""
function rom_reconstruct(a,ϕ)
    u = ϕ*a
end

"""
    rom_project(u,ϕ)

    Project FOM velocity u onto ROM basis ϕ to obtain ROM coefficient vector a
"""
function rom_project(u,ϕ)
    a = ϕ'*u
end
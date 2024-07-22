using LinearAlgebra

function tuple2vec(u,setup)
    Iu = setup.grid.Iu
    sx, sy = u
    [sx[Iu[1]][:]; sy[Iu[2]][:]]
end

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
end

export tuple2vec
export vec2tuple

function compute_POD_basis(snapshots,r,setup)
    vecshots = map(snapshots.u) do u
        tuple2vec(u,setup)
    end

    snapmat = stack(vecshots)

    svd_ = svd(snapmat)

    ϕ = svd_.U[:,1:r]

    ϕ,svd_,snapmat
end

function rom_reconstruct(a,ϕ)
    u = ϕ*a
end

function rom_project(u,ϕ)
    a = ϕ'*u
end
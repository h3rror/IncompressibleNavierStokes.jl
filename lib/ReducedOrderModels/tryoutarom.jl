using ReducedOrderModels
ROM = ReducedOrderModels

using IncompressibleNavierStokes
INS = IncompressibleNavierStokes

# # Shear layer - 2D
#
# Shear layer example.

using CairoMakie
using GLMakie
# using IncompressibleNavierStokes

using LinearAlgebra

# Reynolds number
Re = 2000.0

# A 2D grid is a Cartesian product of two vectors
n = 128
lims = 0.0, 2π
x = LinRange(lims..., n + 1)
y = LinRange(lims..., n + 1)
plotgrid(x, y)

# Build setup and assemble operators
setup = Setup(x, y; Re);

# Initial conditions: We add 1 to u in order to make global momentum
# conservation less trivial
d = π / 15
e = 0.05
U1(y) = y ≤ π ? tanh((y - π / 2) / d) : tanh((3π / 2 - y) / d)
## U1(y) = T(1) + (y ≤ π ? tanh((y - T(π / 2)) / d) : tanh((T(3π / 2) - y) / d))
ustart = create_initial_conditions(setup, (dim, x, y) -> dim() == 1 ? U1(y) : e * sin(x));

snapshots = ReducedOrderModels.create_snapshots(;
    setup,
    ustart,
    Δt = 0.01,
    n_snap = 800,
)

r = 10
ϕ,svd_,snapmat = ROM.compute_POD_basis(snapshots,r,setup)

CairoMakie.activate!()
scatter(svd_.S; axis = (; yscale = log10))

astart = ROM.rom_project(tuple2vec(ustart,setup),ϕ)
a,t = ROM.rom_timestep_loop(ϕ;setup,nstep = 10,astart)

div_ = INS.divergence(vec2tuple(ROM.rom_reconstruct(a,ϕ),setup),setup)
maximum(div_[:])
norm(div_[:])

ROM.convection(ustart,ustart,setup)
ROM.convection(ustart,ustart,setup) == INS.convection(ustart,setup)

ustart_vec = tuple2vec(ustart,setup)
tuple2vec(ustart,setup)'*tuple2vec(INS.convection(ustart,setup),setup)
tuple2vec(ustart,setup)'*tuple2vec(ROM.convection(ustart,ustart,setup),setup)

u2 = vec2tuple(tuple2vec(ustart,setup) .^2,setup)

tuple2vec(ustart,setup)'*tuple2vec(ROM.convection(u2,ustart,setup),setup)
tuple2vec(ustart,setup)'*tuple2vec(ROM.convection(ustart,u2,setup),setup)

u1 = vec2tuple(ϕ[:,1],setup)
u2 = vec2tuple(ϕ[:,2],setup)

tuple2vec(u1,setup)'*tuple2vec(ROM.convection(u2,u1,setup),setup)
tuple2vec(u1,setup)'*tuple2vec(ROM.convection(u1,u2,setup),setup)
tuple2vec(u1,setup)'*tuple2vec(ROM.convection(u1,u1,setup),setup)

C_r2,C_r1,y_C = ROM.rom_convection_operator(ϕ,setup)

ROM_setup = ROM.rom_operators(ϕ, setup)

nstep = 10
Δt = 0.01
ROM.rom_timestep_loop_efficient(ROM_setup; setup, nstep, astart, Δt)

ROM.rom_diffusion(a, ROM_setup)
b = a
ROM.rom_convection(a, b, ROM_setup)

rs = ROM.ROMstate(a)



snapshots.u[1]
snapshots.u[1][1]
snapshots.u[1][1][setup.grid.Iu[1]]
snapshots.u[1][2][setup.grid.Iu[2]]
snapshots.u[1][1][setup.grid.Iu[1]][:]
vecshot = [
    snapshots.u[1][1][setup.grid.Iu[1]][:];
    snapshots.u[1][2][setup.grid.Iu[2]][:]
]

v = tuple2vec(snapshots.u[1], setup)
vec2tuple(v, setup)[2]

vecshots = map(snapshots.u) do u
    Iu = setup.grid.Iu
    sx, sy = u
    [sx[Iu[1]][:]; sy[Iu[2]][:]]
end

# map(f, [1, 2, 3]) = [f(1), f(2), f(3)]

snapmat = stack(vecshots)

using LinearAlgebra
U, S, V = svd(snapmat)
s = svd(snapmat)

U
S
V
CairoMakie.activate!()
scatter(S; axis = (; yscale = log10))


n
U[:, 1]

u = vec2tuple(U[:, 10], setup)

u[2] |> heatmap

# Plot vorticity
for i = 1:50
    u = vec2tuple(U[:, i], setup)
    fig = fieldplot((; u, t = 0.0, temp = nothing); setup, fieldname = :vorticity)
    display(fig)
    sleep(0.1)
end

# Solve unsteady problem
state, outputs = solve_unsteady(;
    setup,
    ustart,
    tlims = (0.0,  8.0),
    Δt = 0.01,
    processors = (
        rtp = realtimeplotter(;
            setup,
            plot = fieldplot,
            ## plot = energy_history_plot,
            ## plot = energy_spectrum_plot,
            nupdate = 10,
            displayupdates = true,
        ),
        ## anim = animator(; setup, path = "$output/vorticity.mkv", nupdate = 20),
        ## vtk = vtk_writer(; setup, nupdate = 10, dir = output, filename = "solution"),
        ## field = fieldsaver(; setup, nupdate = 10),
        log = timelogger(; nupdate = 1),
    ),
);

# ## Post-process
#
# We may visualize or export the computed fields

outputs.rtp

# Plot pressure
fieldplot(state; setup, fieldname = :pressure)

# Plot velocity
fieldplot(state; setup, fieldname = :velocitynorm)

# Plot vorticity
fieldplot(state; setup, fieldname = :vorticity)
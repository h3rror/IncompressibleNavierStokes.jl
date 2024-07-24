using ReducedOrderModels
using IncompressibleNavierStokes

Re = 2000.0
n = 128
lims = 0.0, 2π
x = LinRange(lims..., n + 1)
y = LinRange(lims..., n + 1)
setup = Setup(x, y; Re);
d = π / 15
e = 0.05
U1(y) = y ≤ π ? tanh((y - π / 2) / d) : tanh((3π / 2 - y) / d)
## U1(y) = T(1) + (y ≤ π ? tanh((y - T(π / 2)) / d) : tanh((T(3π / 2) - y) / d))
ustart = create_initial_conditions(setup, (dim, x, y) -> dim() == 1 ? U1(y) : e * sin(x));



# Functions that are basically copies of function in IncompressibleNavierStokes.jl,
# but with slight modifications.
# 
# Ideally, such functions are only introduced if multiple dispatch or other ways of reusing 
# existing code are not possible

# Maybe this function should be integrated into INS via a pull-request?

using KernelAbstractions

struct Offset{D} end # better: reuse INS code

@inline (::Offset{D})(α) where {D} = CartesianIndex(ntuple(β -> β == α ? 1 : 0, D)) # better: reuse INS code

"""
    convection!(F, u, v, setup)

    Modification of IncompressibleNavierStokes.convection!(F, u, setup) that distinguishes
    between convected velocity u and convecting velocity v,
    i.e. the inner product of u and convection(u,v,setup) is 0 if v is divergence-free
"""
function convection!(F, u, v, setup)
    # @warn("check with Syver what I did here. I have no idea")
    (; grid, workgroupsize) = setup
    (; dimension, Δ, Δu, Nu, Iu, A) = grid
    D = dimension()
    e = Offset{D}()
    @kernel function conv!(F, u, ::Val{α}, ::Val{βrange}, I0) where {α,βrange}
        I = @index(Global, Cartesian)
        I = I + I0
        # for β = 1:D
        KernelAbstractions.Extras.LoopInfo.@unroll for β in βrange
            Δuαβ = α == β ? Δu[β] : Δ[β]

            # Half for u[α], (reverse!) interpolation for u[β]
            # Note:
            #     In matrix version, uses
            #     1*u[α][I-e(β)] + 0*u[α][I]
            #     instead of 1/2 when u[α][I-e(β)] is at Dirichlet boundary.
            uαβ1 = (u[α][I-e(β)] + u[α][I]) / 2
            uαβ2 = (u[α][I] + u[α][I+e(β)]) / 2
            vβα1 =
                A[β][α][2][I[α]-(α==β)] * v[β][I-e(β)] +
                A[β][α][1][I[α]+(α!=β)] * v[β][I-e(β)+e(α)]
            vβα2 = A[β][α][2][I[α]] * v[β][I] + A[β][α][1][I[α]+1] * v[β][I+e(α)]

            # # Half
            # uαβ1 = (u[α][I-e(β)] + u[α][I]) / 2
            # uβα1 = u[β][I-e(β)] / 2 + u[β][I-e(β)+e(α)] / 2
            # uαβ2 = (u[α][I] + u[α][I+e(β)]) / 2
            # uβα2 = u[β][I] / 2 + u[β][I+e(α)] / 2

            # # Interpolation
            # uαβ1 = A[α][β][2][I[β]-1] * u[α][I-e(β)] + A[α][β][1][I[β]] * u[α][I]
            # uβα1 =
            #     A[β][α][2][I[α]-(α==β)] * u[β][I-e(β)] +
            #     A[β][α][1][I[α]+(α!=β)] * u[β][I-e(β)+e(α)]
            # uαβ2 = A[α][β][2][I[β]] * u[α][I] + A[α][β][1][I[β]+1] * u[α][I+e(β)]
            # uβα2 = A[β][α][2][I[α]] * u[β][I] + A[β][α][1][I[α]+1] * u[β][I+e(α)]

            F[α][I] -= (uαβ2 * vβα2 - uαβ1 * vβα1) / Δuαβ[I[β]]
        end
    end
    for α = 1:D
        I0 = first(Iu[α])
        I0 -= oneunit(I0)
        conv!(get_backend(F[1]), workgroupsize)(F, u, Val(α), Val(1:D), I0; ndrange = Nu[α])
    end
    F
end

convection(u, v, setup) = convection!(zero.(u), u, v, setup)



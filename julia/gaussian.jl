include("prox.jl")

using LinearAlgebra
using SparseArrays
using Base.Threads
using Distributions

function deviance(::Normal, residual, y)
    norm(residual)^2
end

function primal(::Normal, residual, β, λ)
    0.5*norm(residual)^2 + λ*norm(β, 1)
end

function dual(::Normal, y, residual)
    dot(residual, y) - 0.5*norm(residual)^2
    # or 0.5*norm(y)^2 - 0.5*norm(X*β)^2
end

# function residual(::Normal, Xβ, y)
#     y .- Xβ
# end

function scaledDual(::Normal, y, residual, λ, dual_scale)
    if dual_scale == 0
        return 0.0
    end

    α = λ/dual_scale
    α*dot(residual, y) - 0.5*(α*norm(residual))^2
end

function gradient(::Normal, X, residual)
    -X'*residual
end

function hessian(::Normal, X::Matrix{Float64}, X_mean_scaled, w, indices, standardize)
    Symmetric(X[:, indices]'X[:, indices])
end

function hessian(::Normal, X::SparseMatrixCSC{Float64}, X_mean_scaled, w, indices, standardize)
    H = Matrix(X[:, indices]'X[:, indices])

    if standardize
        H .-= size(X, 1)*(X_mean_scaled[indices]*X_mean_scaled[indices]')
    end

    Symmetric(H)
end

function hessianUpperrightBlock(::Normal,
                                Xa::Matrix{Float64},
                                Xb::Matrix{Float64},
                                Xa_mean_scaled,
                                Xb_mean_scaled,
                                w,
                                standardize)
    Xa'Xb
end

function hessianUpperrightBlock(::Normal,
                                Xa::SparseMatrixCSC{Float64},
                                Xb::SparseMatrixCSC{Float64},
                                Xa_mean_scaled,
                                Xb_mean_scaled,
                                w,
                                standardize)
    H = Matrix(Xa'Xb)

    if standardize
        H .-= size(Xa, 1)*(Xa_mean_scaled*Xb_mean_scaled')
    end

    H
end

function updateResidual!(::Normal,
                         residual::Vector{T},
                         y::Vector{T},
                         X::Matrix{T},
                         β::Vector{T},
                         X_mean_scaled::Vector{T},
                         indices,
                         standardize::Bool) where {T <: Float64}
    residual .= y .- X[:, indices]*β[indices]
end

function updateResidual!(::Normal,
                         residual::Vector{T},
                         y::Vector{T},
                         X::SparseMatrixCSC{T, Int64},
                         β::Vector{T},
                         X_mean_scaled::Vector{T},
                         indices,
                         standardize::Bool) where {T <: Float64}

    residual .= y .- Array(X[:, indices]*β[indices])

    if standardize
        residual .+= dot(β[indices], X_mean_scaled[indices])
    end
end

function fitNullModel(family::Normal, y, intercept)
    mean(y)
end

function safeScreening!(::Normal,
                        X,
                        X_mean_scaled,
                        residual,
                        c,
                        XTcenter,
                        β,
                        screened,
                        X²sums,
                        r_screen,
                        standardize)

    for j in findall(screened)
        r_normX_j = r_screen * sqrt(X²sums[j])

        if r_normX_j >= 1
            continue
        end

        if abs(XTcenter[j]) + r_normX_j < 1
            if β[j] != 0
                # update residual
                residual .+= β[j] * X[:, j]
                if issparse(X) && standardize
                    residual .-= X_mean_scaled[j] * β[j]
                end
                β[j] = 0
            end

            screened[j] = false
        end
    end
end

function lasso(family::Normal,
               X::AbstractMatrix{T},
               X²sums::Vector{T},
               X_mean_scaled::Vector{T},
               y::Vector{T},
               β::Vector{T},
               c::Vector{T},
               residual::Vector{T},
               λ::T,
               λ_max::T,
               screened::BitVector,
               null_deviance::Float64;
               standardize::Bool = true,
               screening_type = "hessian",
               maxit::Int64 = Int64(1e6),
               tol_decr::T = 1e-7,
               tol_gap::T = 1e-4,
               tol_infeas::T = 1e-4,
               verbosity::Int = 0) where {T <: Float64}

    n, p = size(X)

    it = 0

    X_is_sparse = issparse(X)

    primal_value = 0.0
    dual_value = 0.0

    XTcenter = Vector{Float64}(undef, p)
    dual_scale = λ

    screen_interval = 10

    if screening_type in ["gap_safe"]
        screened .= true
    end

    n_screened = 0
    screened_set = findall(screened)

    if !isempty(screened_set)
        updateResidual!(family, residual, y, X, β, X_mean_scaled, screened_set, standardize)

        while it < maxit
            it += 1

            if verbosity >= 2
                println("\t\titer: $(it)")
            end

            if screening_type in ["gap_safe"] && (mod(it, screen_interval) == 0 || it == 1)
                if it > 1
                    # need to update correlation for all screened predictors
                    updateResidual!(family,
                                    residual,
                                    y,
                                    X,
                                    β,
                                    X_mean_scaled,
                                    screened_set,
                                    standardize)
                end
                updateCorrelation!(c, X, X_mean_scaled, residual, screened_set, standardize)

                dual_scale = max(λ, maximum(abs.(c)))

                primal_value = primal(family, residual, β[screened_set], λ)
                dual_value = scaledDual(family, y, residual, λ, dual_scale)
                duality_gap = abs(primal_value - dual_value)

                max_infeasibility = if λ > 0 maximum(abs.(c[screened_set]) .- λ) else 0.0 end

                if duality_gap <= tol_gap*null_deviance && max_infeasibility <= tol_infeas*λ_max
                    break
                end

                if screening_type == "gap_safe"
                    for j in 1:p
                        XTcenter[j] = c[j] / dual_scale
                    end
                    r_screen = sqrt(duality_gap) / λ
                end

                safeScreening!(Normal(),
                              X,
                              X_mean_scaled,
                              residual,
                              c,
                              XTcenter,
                              β,
                              screened,
                              X²sums,
                              r_screen,
                              standardize)
                screened_set = findall(screened)
            end

            n_screened += length(screened_set)

            primal_value_old = primal(family, residual, β[screened_set], λ)

            for j in screened_set
                β_j_old = β[j]
                c[j] = dot(X[:, j], residual)
                if standardize && issparse(X)
                    c[j] -= X_mean_scaled[j]*sum(residual)
                end
                β[j] = prox(c[j] + X²sums[j]*β_j_old, λ)/X²sums[j]

                if β_j_old != β[j]
                    βdiff = β[j] - β_j_old
                    residual .-= X[:, j] * βdiff

                    if X_is_sparse && standardize
                        residual .+= X_mean_scaled[j] * βdiff
                    end
                end
            end

            primal_value = primal(family, residual, β[screened_set], λ)
            primal_value_change = primal_value - primal_value_old

            if verbosity >= 2
                println("\t\tprimal_value_change=$(primal_value_change)")
            end

            if abs(primal_value_change) <= tol_gap*primal_value
                dual_value = dual(family, y, residual)
                duality_gap = abs(primal_value - dual_value)

                updateCorrelation!(c, X, X_mean_scaled, residual, screened_set, standardize)

                max_infeasibility = if λ > 0 maximum(abs.(c[screened_set]) .- λ) else 0.0 end

                if verbosity >= 2
                    println("\t\tmax_infeas: $(max_infeasibility/λ_max)")
                    println("\t\tduality_gap: $(duality_gap/null_deviance)")
                end

                if max_infeasibility <= tol_infeas*λ_max && duality_gap <= tol_gap*null_deviance
                    break
                end
            end
        end
    else
        fill!(β, 0)
    end

    avg_screened = n_screened/it

    (passes = it,
     primal = primal_value,
     dual = dual_value,
     dual_scale = dual_scale,
     avg_screened = avg_screened)
end

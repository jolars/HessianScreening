include("prox.jl")
include("updateLinearPredictor.jl")

using LinearAlgebra

const P_MIN = 1e-9
const P_MAX = 1.0 - P_MIN

# sigmoid function
σ(Xβ) = 1.0 / (1.0 + exp(-Xβ))

function primal(::Binomial, Xβ, β, y, λ)
    -sum(Xβ[y .== 1]) + sum(log1p.(exp.(Xβ))) + λ*norm(β, 1)
end

function dual(::Binomial, pr)
    -sum(pr .* log.(pr) .+ (1 .- pr) .* log.(1 .- pr))
end

function scaledDual(::Binomial, y, residual, λ, dual_scale)
    α = λ / dual_scale

    pr = y .- α .* residual
    clamp!(pr, P_MIN, P_MAX)

    -sum(pr .* log.(pr) .+ (1 .- pr) .* log.(1 .- pr))
end

# function residual(::Binomial, Xβ, y)
#     y .- clamp(σ.(Xβ), P_MIN, P_MAX)
# end

function deviance(::Binomial, residual::Vector{T}, y::Vector{T}) where {T <: Float64}
    pr = clamp.(y .- residual, P_MIN, P_MAX)
    -2*sum(pr .* log.(pr) .+ (1.0 .- pr) .* log.(1.0 .- pr))
end

function hessian(::Binomial, X::Matrix{Float64}, X_mean_scaled, w, indices, standardize)
    Xi = X[:, indices]
    Symmetric(Xi'*Diagonal(w)*Xi)
end

function hessian(::Binomial, X::SparseMatrixCSC{Float64}, X_mean_scaled, w, indices, standardize)
    Xi = X[:, indices]

    D = Diagonal(w)

    H = Xi'*D*Xi

    if standardize
        XmDX = X_mean_scaled[indices]*sum(D*Xi, dims = 1)
        H += sum(w)*X_mean_scaled[indices]*X_mean_scaled[indices]' - XmDX - XmDX'
    end

    Symmetric(H)
end


function hessianUpperrightBlock(::Binomial,
                                Xa::Matrix{Float64},
                                Xb::Matrix{Float64},
                                Xa_mean_scaled,
                                Xb_mean_scaled,
                                w,
                                standardize)
    Xa'*Diagonal(w)*Xb
end

function hessianUpperrightBlock(::Binomial,
                                Xa::SparseMatrixCSC{Float64},
                                Xb::SparseMatrixCSC{Float64},
                                Xa_mean_scaled,
                                Xb_mean_scaled,
                                w,
                                standardize)
    H = Xa'*D(w)*Xb

    if standardize
        XamDXb = Xa_mean_scaled*sum(D*Xb, dims = 1)
        XbmDXa = Xb_mean_scaled*sum(D*Xa, dims = 1)
        H += sum(d)*Xa_mean_scaled*Xb_mean_scaled' - XbmDXa' - XamDXb
    end

    H
end

function updateResidual!(::Binomial,
                         residual::Vector{T},
                         y::Vector{T},
                         X::SparseMatrixCSC{T, Int64},
                         β::Vector{T},
                         X_mean_scaled::Vector{T},
                         indices,
                         standardize::Bool) where {T<:Float64}
    Xβ = Array(X[:, indices]*β[indices])
    if standardize
        Xβ .+= dot(β[indices], X_mean_scaled[indices])
    end
    pr = σ.(Xβ)
    clamp!(pr, P_MIN, P_MAX)
    residual .= y .- pr
end

function updateResidual!(::Binomial,
                         residual::Vector{T},
                         y::Vector{T},
                         X::Matrix{T},
                         β::Vector{T},
                         X_mean_scaled::Vector{T},
                         indices,
                         standardize::Bool) where {T<:Float64}
    pr = σ.(X[:, indices]*β[indices])
    clamp!(pr, P_MIN, P_MAX)
    residual .= y .- pr
end

function updateResidualsProbabilitiesExpβ!(residual, expXβ, pr, Xβ)
    expXβ    .= exp.(Xβ)
    pr       .= clamp.(expXβ ./ (1.0 .+ expXβ), P_MIN, P_MAX)
    residual .= y .- pr
end

function fitNullModel(family::Binomial, y, intercept)
    if (intercept)
        pr::Float64 = clamp(mean(y), P_MIN, P_MAX)
        return log(pr / (1 - pr))
    else
        return 0.0
    end
end

function safeScreening!(::Binomial,
                        X,
                        Xβ,
                        X_mean_scaled,
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
                # update linear predictor
                Xβ .-= β[j]*X[:, j]
                if issparse(X) && standardize
                    Xβ .+= X_mean_scaled[j]*β[j]
                end
                β[j] = 0
            end

            screened[j] = false
        end
    end
end

function lasso(family::Binomial,
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

    it = 0

    n, p = size(X)

    primal_value = 0.0
    dual_value = 0.0
    duality_gap = Inf
    dual_scale = λ

    screen_interval = 10

    if screening_type in ["gap_safe"]
        screened .= true
    end

    n_screened = 0
    screened_set = findall(screened)

    if !isempty(screened_set)
        Xβ    = zeros(n)
        expXβ = zeros(n)
        pr    = zeros(n)

        XTcenter = Vector{Float64}(undef, p)

        updateLinearPredictor!(Xβ, X, X_mean_scaled, β, screened_set, standardize)
        updateResidualsProbabilitiesExpβ!(residual, expXβ, pr, Xβ)

        while it < maxit
            it += 1

            if verbosity >= 2
                println("\t\titer: $(it)")
            end

            if screening_type in ["gap_safe"] && (mod(it, screen_interval) == 0 || it == 1)

                if it > 1
                    # already updated for first iteration
                    updateLinearPredictor!(Xβ, X, X_mean_scaled, β, screened_set, standardize)
                    updateResidualsProbabilitiesExpβ!(residual, expXβ, pr, Xβ)
                end

                # need to update correlation for all screened predictors
                updateCorrelation!(c, X, X_mean_scaled, residual, screened_set, standardize)

                dual_scale = max(λ, maximum(abs.(c)))

                primal_value = sum(log1p.(expXβ)) - sum(Xβ[y .== 1]) + λ*norm(β[screened_set], 1)
                dual_value = scaledDual(family, y, residual, λ, dual_scale)
                duality_gap = abs(primal_value - dual_value)

                if screening_type == "gap_safe"
                    for j in screened_set
                        XTcenter[j] = c[j] / dual_scale
                    end

                    r_screen = sqrt(2 * duality_gap) / (2 * λ)
                end

                safeScreening!(Binomial(),
                                X,
                                Xβ,
                                X_mean_scaled,
                                c,
                                XTcenter,
                                β,
                                screened, X²sums, r_screen, standardize)

                screened_set = findall(screened)
                updateResidualsProbabilitiesExpβ!(residual, expXβ, pr, Xβ)
            end

            n_screened += length(screened_set)

            β_screened_old = copy(β[screened_set])
            primal_value_old = sum(log1p.(expXβ)) - sum(Xβ[y .== 1]) + λ*norm(β_screened_old, 1)
            dual_value_old = dual(family, pr)

            for (k, j) in enumerate(screened_set)
                β_j_old = β[j]
                c[j] = dot(X[:, j], residual)
                if standardize && issparse(X)
                    c[j] -= X_mean_scaled[j]*sum(residual)
                end

                w = pr .* (1 .- pr)
                hes = dot(X[:, j].^2, w)

                if issparse(X) && standardize
                    hes += sum(w)*X_mean_scaled[j]^2 - 2*dot(X[:, j], w)*X_mean_scaled[j]
                end

                hes += 1e-12
                β[j] = prox(c[j] + hes*β_j_old, λ) / hes

                if β_j_old != β[j]
                    βdiff = (β[j] - β_j_old)
                    Xβ .+= X[:, j] * βdiff

                    if issparse(X) && standardize
                        Xβ .-= X_mean_scaled[j]*βdiff
                    end

                    updateResidualsProbabilitiesExpβ!(residual, expXβ, pr, Xβ)
                end
            end

            primal_value = sum(log1p.(expXβ)) - sum(Xβ[y .== 1]) + λ*norm(β[screened_set], 1)
            dual_value = dual(Binomial(), pr)

            t = 0.999
            β_screened = β[screened_set]

            line_it = 0

            while primal_value >= primal_value_old && dual_value <= dual_value_old && line_it < 5
                # if we haven't progressed in either dual or primal solution, use
                # momentum from previous estimates to take a step

                line_it += 1
                if verbosity >= 2
                    println("\t\tno progress; taking a momentum step")
                end

                β[screened_set] = (1.0 - t) .* β_screened_old + t .* β_screened

                updateLinearPredictor!(Xβ, X, X_mean_scaled, β, screened_set, standardize)
                updateResidualsProbabilitiesExpβ!(residual, expXβ, pr, Xβ)

                primal_value = sum(log1p.(expXβ)) - sum(Xβ[y .== 1]) + λ*norm(β[screened_set], 1)
                dual_value = dual(Binomial(), pr)

                t *= 0.5
            end

            primal_value_change = primal_value - primal_value_old

            if verbosity >= 2
                println("\t\tprimal_value_change=$(primal_value_change)")
            end

            if abs(primal_value_change) <= tol_gap*primal_value
                dual_value = dual(family, pr)
                duality_gap = abs(primal_value - dual_value)

                updateCorrelation!(c, X, X_mean_scaled, residual, screened_set, standardize)
                max_infeas = if λ > 0 maximum(abs.(c[screened]) .- λ) else 0 end

                if verbosity >= 2
                    println("\t\tmax_infeas: $(max_infeas/λ_max)")
                    println("\t\tduality_gap: $(duality_gap/null_deviance)")
                end

                if max_infeas <= tol_infeas*λ_max && duality_gap <= tol_gap*null_deviance
                    break
                end
            end
        end
    else
        fill!(β, 0)
    end

    if it == maxit
        @warn "maximum number of iterations ($maxit) reached in CD loop"
    end

    avg_screened = n_screened/it

    (passes = it,
     primal = primal_value,
     dual = dual_value,
     dual_scale = dual_scale,
     avg_screened = avg_screened)
end

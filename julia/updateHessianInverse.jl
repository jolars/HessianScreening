using LinearAlgebra
using SparseArrays

include("gaussian.jl")
include("logistic.jl")

function findDuplicates(D, activate)
    originals = Int64[]
    duplicates = Int64[]

    for i in 1:(size(D, 1) - 1)
        if !(activate[i] in duplicates)
            for j in (i + 1):size(D, 1)
                if sqrt(D[j, j] * D[i, i]) == abs(D[i, j])
                    push!(originals, activate[i])
                    push!(duplicates, activate[j])
                end
            end
        end
    end

    originals, duplicates
end

function updateHessianInverse(family::UnivariateDistribution,
                              H::Symmetric{Float64, Matrix{Float64}},
	                          H⁻::Symmetric{Float64, Matrix{Float64}},
                              X::AbstractMatrix{T},
                              X_mean_scaled::Vector{T},
                              DX::AbstractMatrix{T},
                              w::Vector{T},
                              active_set::Vector{Int64},
                              active_prev_set::Vector{Int64};
                              verify::Bool = false,
                              logistic_hessian_updates::String = ["full", "approx", "upperbound"][1],
                              verbosity::Int64 = 0,
                              reset_hessian::Bool = false,
                              standardize::Bool = true) where {T<:Float64}
    if verbosity >= 1
        println("\tupdating Hessian inverse")
    end

    n, p = size(X)

    if family isa Binomial && logistic_hessian_updates == "approx" && !reset_hessian
        X = DX
    end

    deactivate = setdiff(active_prev_set, active_set)
    activate   = setdiff(active_set, active_prev_set)

    new_originals = Int64[]
    new_duplicates = Int64[]

    reset_hessian = reset_hessian || (logistic_hessian_updates == "full" && family isa Binomial)

    if !isempty(deactivate) && !reset_hessian
        if family isa Normal || logistic_hessian_updates in ["approx", "upperbound"]
            # update matrix inverse by removing effect from deactivated predictors
            if verbosity >= 1
                println("\t\tdropping deactivated predictors for inverse n = $(length(deactivate))")
            end

            remain = intersect(active_prev_set, active_set)

            new_perm = [remain; deactivate]
            keep = [i for i in 1:length(active_prev_set) if active_prev_set[i] in active_set]
            drop = [i for i in 1:length(active_prev_set) if !(active_prev_set[i] in active_set)]

            H⁻kd = H⁻[keep, drop]
            #Λ, Q = eigen(H⁻[drop, drop])
            #tol = length(drop)*eps(Float64)*10
            #index = Λ .> tol
            #Λ[index] = 1 ./ Λ[index]
            #Λ[.!index] .= 0
            #H⁻inv = Symmetric(Q * Diagonal(Λ) * Q')
            H⁻dd = family isa Normal ? cholesky(H⁻[drop, drop]) : bunchkaufman(H⁻[drop, drop])
            H⁻ = Symmetric(H⁻[keep, keep] - H⁻kd*(H⁻dd \ H⁻kd'))
            active_prev_set = copy(remain)

            H = Symmetric(H[keep, keep])
        end
    end

    if !isempty(activate)
        # update matrix inverse with new predictors
        if verbosity >= 1
            println("\t\tadding newly activated predictors to inverse n = $(length(activate))")
        end

        X_active_prev = X[:, active_prev_set]
        X_new = X[:, activate]

        D = hessian(Normal(),
                    X_new,
                    X_mean_scaled[activate],
                    w,
                    1:length(activate),
                    standardize)

        if family isa Binomial && logistic_hessian_updates == "upperbound"
            D *= 0.25
        end

        # find duplicated columns and deactivate all but one as well as
        # store duplicated groups
        # duplicated columns always enter together, so no risk of overlap

        new_originals, new_duplicates = findDuplicates(D, activate)

        if verbosity > 0 && !isempty(new_duplicates)
            println("\tfound (and droppped) duplicates $(new_duplicates)")
        end

        if !isempty(new_duplicates)
            activate = setdiff(activate, new_duplicates)
            active_set = setdiff(active_set, new_duplicates)
            X_new = X[:, activate]

            D = hessian(Normal(),
                        X_new,
                        X_mean_scaled[activate],
                        w,
                        1:length(activate),
                        standardize)

            if family isa Binomial && logistic_hessian_updates == "upperbound"
                D *= 0.25
            end
        end

        if family isa Normal || logistic_hessian_updates in ["upperbound", "approx"]
            A  = H
            A⁻ = H⁻

            B = hessianUpperrightBlock(Normal(),
                                       X_active_prev,
                                       X_new,
                                       X_mean_scaled[active_prev_set],
                                       X_mean_scaled[activate],
                                       w,
                                       standardize)

            if family isa Binomial && logistic_hessian_updates == "upperbound"
                B *= 0.25
            end

            C = B'
            A⁻B = A⁻*B
            S = Symmetric(D - C*A⁻B)
            Λ, Q = eigen(S)
            # D[diagind(D)] .+= 1e-4*n
            if minimum(Λ) < 1e-4*n
                D[diagind(D)] .+= 1e-4*n
                Λ .+= 1e-4*n
            end
            Λ = 1 ./ Λ
            S⁻ = Symmetric(Q * Diagonal(Λ) * Q')
            A⁻BS⁻ = A⁻B*S⁻

            H = Symmetric([A B;
                        C D])

            H⁻ = Symmetric([(A⁻BS⁻*C + I)*A⁻    -A⁻BS⁻;
                            -(A⁻BS⁻)'               S⁻])

        end
    end

    if reset_hessian
        H = hessian(family, X, X_mean_scaled, w, active_set, standardize)
        H[diagind(H)] .+= 1e-4*n
        Λ, Q = eigen(H)
        H⁻ = Symmetric(Q * Diagonal(1 ./ Λ) * Q')
    end

	if verify
        hessian_inverse_error = norm(H - H*H⁻*H, Inf)

        if hessian_inverse_error >= 0.01
            throw("inverse matrix computation is not correct")
        end
    end

    return H, H⁻, new_originals, new_duplicates
end

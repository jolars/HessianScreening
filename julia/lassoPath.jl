using LinearAlgebra
using Statistics
using SparseArrays
using Base.Threads
using Distributions
using FileIO

include("checkStoppingConditions.jl")
include("checkViolations.jl")
include("colNormsSquared.jl")
include("gaussian.jl")
include("getNextLambda.jl")
include("kktCheck.jl")
include("logistic.jl")
include("screenPredictors.jl")
include("standardize.jl")
include("updateCorrelation.jl")
include("updateGradientOfCorrelation.jl")
include("updateHessianInverse.jl")
include("updateLinearPredictor.jl")

function lassoPath(X::AbstractMatrix{T},
                   y::Vector{T},
                   family::UnivariateDistribution = Normal();
                   intercept::Bool = true,
                   standardize::Bool = true,
                   screening_type::String = "hessian",
                   logistic_hessian_updates::String = ["full", "approx", "upperbound"][1],
                   copyy::Bool = true,
                   copyX::Bool = true,
                   hessian_warm_starts::Bool = true,
                   active_warm_start::Bool = true,
                   maxit::Int64 = Int64(1e6),
                   tol_decr::T = 1e-4,
                   tol_gap::T = 1e-4,
                   tol_infeas::T = 1e-4,
                   path_length::Int64 = 100,
                   γ::T = 0.01,
                   η::T = 1.5,
                   verify::Bool = false,
                   verbosity::Int64 = 0) where {T <: Float64}

    n, p = size(X)

    if copyX && standardize
        X = copy(X)
    end

    if copyy
        y = copy(y)
    end

    if family isa Binomial
        if sort(unique(y)) != [0.0, 1.0]
            throw("response is not in {0,1}")
        end
    end

    if family isa Binomial && screening_type == "edpp"
        throw("EDPP screening cannot be used with logistic regression")
    end

    if size(X, 1) != length(y)
        DimensionMismatch("number of observations in y and X do not match")
    end

    standardizeY!(family, y)
    X_mean, X_sd = standardizeX!(X, standardize)
    X_mean_scaled = X_mean ./ X_sd
    X_mean_sum = sum(X_mean)

    β = zeros(p)
    residual = copy(y)
    c = zeros(p)

    updateResidual!(family, residual, y, X, β, X_mean_scaled, Int64[], standardize)
    updateCorrelation!(c, X, X_mean_scaled, residual, 1:p, standardize)

    c_pred = copy(c)
    ∇c = zeros(p)

    # get λ corresponding to entry of first predictor
    λ_min_ratio = if (n < p) 0.01 else 0.0001 end
    λ_max = maximum(abs.(c))
    λ_min = λ_max * λ_min_ratio
    λfixedpath = exp.(range(log(λ_max), log(λ_min), length = path_length))
    λ_minstep = (λfixedpath[path_length - 1] - λfixedpath[path_length])/2
    λs = screening_type == "hessian_adaptive" ? [λ_max] : copy(λfixedpath)
    λ_prev = λ_max
    λ = λ_max

    β = zeros(p)
    βs = zeros((p, 0))

    primals = Float64[]
    duals = Float64[]
    dev_ratios = Float64[]

    dual_scale = λ_max

    γs              = Float64[]
    violations_list = Int64[]
    refits          = Int64[]
    passes          = Int64[]
    n_active        = Int64[]
    n_screened      = Float64[]
    n_strong        = Int64[]
    n_new_active    = Int64[]

    X²sums = Vector{Float64}(undef, p)

    if family isa Normal || screening_type == "gap_safe"
        if standardize
            X²sums = repeat([Float64(n)], p)
        else
            X²sums = colNormsSquared(X)
        end
    end

    active = falses(p)

    # keep track of duplicated columns that need to be deactivated to keep problem stable
    originals  = Int64[]
    duplicates = Int64[]

    # identity of first active predictor (or several predictors if there are duplicates)
    # is always available
    first_active = findall(abs.(c) .== λ_max)

    if length(first_active) > 1
        for i in 2:length(first_active)
            push!(originals, first_active[1])
            push!(duplicates, first_active[i])
        end
    end

    active[first_active[1]] = true

    inactive    = .!active
    active_prev = copy(active)
    ever_active = copy(active)
    screened    = copy(active)
    strong      = copy(screened)
    violations  = falses(p)

    # keep track of permutations of active set for hessian updates
    active_set      = findall(active)
    active_set_prev = copy(active_set)

    # desired number of predictors to enter at each step (for hessian adaptive)
    n_target_nonzero = min(sum(inactive), ceil(Int64, n < p ? n/path_length : p/path_length))
    λ_next_mod = 1.0

    DX = Matrix{Float64}(undef, size(X, 1), size(X, 2))
    w = zeros(n)
    w_old = zeros(n)

    if family isa Binomial
        pr = clamp.(y .- residual, P_MIN, P_MAX)
        w = pr .* (1 .- pr)
        w_old = copy(w)

        if logistic_hessian_updates == "approx"
            sqrtw = sqrt.(w)
            Dsq = Diagonal(sqrtw)
            DX = Dsq * X
        end
    end

    H  = hessian(family, X, X_mean_scaled, w, active_set, standardize)
    H⁻ = Symmetric(inv(H))

    # sign of betas
    s = zeros(p)
    s[active] = sign.(c[active])

    # temporary value for hessian updates and warm starts
    H⁻s = H⁻ * s[active]

    null_dev  = deviance(family, residual, y)
    dev       = null_dev
    devs      = [dev]
    dev_ratio = 1.0 - dev/null_dev

    check_kkt = !(screening_type in ["gap_safe"])

    # keep tack of time spent for each iteration
    cd_time           = 0.0
    cd_time_list      = Float64[]
    corr_time         = 0.0
    corr_time_list    = Float64[]
    gradcorr_time     = Float64[]
    hessian_time      = 0.0
    hessian_time_list = Float64[]

    i = 0

    full_time = time()

    while true
        i += 1

        if verbosity >= 1
            println("step: ", i, ", λ: ", λ)
        end

        β_prev = copy(β)
        dev_prev = dev

        # loop solver and kkt checks until convergence
        if verbosity >= 1
            println("\tstarting coordinate descent")
        end

        n_passes     = 0
        n_refits     = 0
        n_violations = 0

        corr_time = 0.0
        cd_time = 0.0
        first_run = true

        while true
            # if i == 43
            #     FileIO.save("debugstuff.jld",
            #                 "active_set", active_set,
            #                 "λ", λ,
            #                 "λ_max", λ_max,
            #                 "β", β,
            #                 "X²sums", X²sums,
            #                 "X_mean_scaled", X_mean_scaled,
            #                 "null_deviance", null_dev,
            #                 "residual", residual,
            #                 "c", c,
            #                 "screened", screened)

            #            #throw("asdf")
            # end

            screening_type_choice = screening_type

            if screening_type == "gap_safe" && first_run
                # first run working strategy once for a better warm start with
                # gap safe method
                screening_type_choice = "working"
            end

            t0 = time()
            n_passes_cur, primal_val, dual_val, dual_scale, avg_screened_cur =
                lasso(
                    family,
                    X,
                    X²sums,
                    X_mean_scaled,
                    y,
                    β,
                    c,
                    residual,
                    λ,
                    λ_max,
                    screened,
                    null_dev,
                    standardize = standardize,
                    screening_type = screening_type_choice,
                    maxit = maxit,
                    tol_decr = tol_decr,
                    tol_gap = tol_gap,
                    tol_infeas = tol_infeas,
                    verbosity = verbosity
                )
            cd_time += time() - t0

            if (first_run && screening_type != "gap_safe") || screening_type_choice == "gap_safe"
                push!(n_screened, avg_screened_cur)
                if verbosity >= 1
                    println("\tscreened predictors: ", avg_screened_cur)
                end
            end

            unscreened = .!screened

            t0 = time()

            if check_kkt
                fill!(violations, false)
                if screening_type in ["strong", "edpp"]
                    check_set = findall(unscreened)
                    updateCorrelation!(c, X, X_mean_scaled, residual, check_set, standardize)
                    kktCheck!(violations, c, λ, screened, check_set)
                elseif screening_type in ["hessian", "hessian_adaptive", "working"]
                    # first check for violations in strong setdiff
                    check_set = setdiff(findall(strong .& unscreened), duplicates)

                    updateCorrelation!(c, X, X_mean_scaled, residual, check_set, standardize)
                    kktCheck!(violations, c, λ, screened, check_set)

                    if !any(violations)
                        # then check for violations among remaining predictors
                        check_set = setdiff(findall(.!strong .& unscreened), duplicates)

                        updateCorrelation!(c, X, X_mean_scaled, residual, check_set, standardize)
                        kktCheck!(violations, c, λ, screened, check_set)
                    end
                end
            end

            if screening_type == "gap_safe"
                # updateCorrelation!(c, X, X_mean_scaled, residual, findall(unscreened), standardize)
                # kktCheck!(violations, c, λ, screened, findall(unscreened))
            end

            corr_time += time() - t0

            n_passes += n_passes_cur
            n_violations += sum(violations)

            if !any(violations) && !(screening_type == "gap_safe" && first_run)
                push!(cd_time_list, cd_time)
                push!(passes, n_passes)
                push!(violations_list, n_violations)
                push!(duals, dual_val)
                push!(primals, primal_val)
                push!(refits, n_refits)

                break
            else
                n_refits += 1
            end

            first_run = false
        end

        # update correlation for screened set
        t0 = time()

        updateCorrelation!(c, X, X_mean_scaled, residual, findall(screened), standardize)

        corr_time += time() - t0
        push!(corr_time_list, corr_time)

        if i > 1
            #s = sign.(β)
            #active = β .!= 0
            active = abs.(abs.(c) .- λ) .<= eps(Float64).^(0.25)
            active = active .| (β .!= 0)
            s[active] = sign.(c[active])
            s[.!active] .= 0
        end

        inactive = .!active

        active_set = [intersect(active_set_prev, findall(active));
                      setdiff(findall(active), active_set_prev)]
        inactive_set = setdiff(1:p, active_set)

        dev = deviance(family, residual, y)
        push!(devs, dev)
        push!(dev_ratios, 1 - dev/null_dev)

        stop_path = checkStoppingConditions(
            i,
            n,
            p,
            length(λs),
            sum(active),
            λ,
            λ_min,
            dev,
            dev_prev,
            null_dev,
            screening_type,
            verbosity
        )

        if stop_path
            push!(hessian_time_list, 0)
            break
        end

        if screening_type in ["hessian", "hessian_adaptive"]
            t0 = time()

            reset_hessian = false

            if family isa Binomial
                if logistic_hessian_updates == "approx"
                    # check for weights, store updates weights and possibly recompute DX for
                    # hessian updates
                    pr = clamp.(y .- residual, P_MIN, P_MAX)
                    w_new = pr .* (1 .- pr)
                    w_diff = w_old - w_new

                    ind = findall(abs.(w_diff) .> 0.2)

                    if verbosity > 0
                        println("\tmaximum_weight = $(maximum(w))")
                        println("\t||weights - old weights|| = $(norm(w_diff))")
                    end

                    if !isempty(ind)
                        if verbosity > 0
                            println("\tNOTE: difference in weights exceeds threshold;",
                                    "resetting hessian for $(length(ind)) observations")
                        end
                        w[ind] = w_new[ind]
                        w_old[ind] = w_new[ind]
                        DX[ind, :] = Diagonal(sqrt.(w[ind])) * X[ind, :]
                        reset_hessian = true
                    end
                elseif logistic_hessian_updates == "full"
                    pr = clamp.(y .- residual, P_MIN, P_MAX)
                    w = pr .* (1 .- pr)
                    reset_hessian = true
                elseif logistic_hessian_updates == "upperbound"
                    w = repeat([0.25], n)
                end
            end

            H, H⁻, new_originals, new_duplicates = updateHessianInverse(
                    family,
                    H,
                    H⁻,
                    X,
                    X_mean_scaled,
                    DX,
                    w,
                    active_set,
                    active_set_prev,
                    verify = verify,
                    logistic_hessian_updates = logistic_hessian_updates,
                    verbosity = verbosity,
                    reset_hessian = reset_hessian,
                    standardize = standardize
                )
            hessian_time = time() - t0

            push!(hessian_time_list, hessian_time)

            # deactivate duplicates
            if !isempty(new_duplicates)
                for (i, orig) in enumerate(unique(new_originals))
                    # for each group of duplicate predictors, assign the sum
                    # of all coefficients to the "original" predictor
                    dups = new_duplicates[new_originals .== orig]

                    β[orig] = sign(c[orig]) * sum(abs.(β[dups]))
                    β[dups] .= 0
                    s[dups] .= 0
                end

                active[new_duplicates] .= false
                active_set = setdiff(active_set, new_duplicates)
                inactive[new_duplicates] .= true
                inactive_set = setdiff(1:p, active_set)
                ever_active[new_duplicates] .= false
            end

            append!(originals, new_originals)
            append!(duplicates, new_duplicates)

            H⁻s = H⁻ * s[active_set]
        else
            push!(hessian_time_list, 0)
        end

        n_active_i = sum(active)
        new_active = length(setdiff(active_set, active_set_prev))
        ever_active = ever_active .| active
        βs = [βs β]

        push!(n_new_active, new_active)
        push!(n_active, n_active_i)
        push!(γs, γ)

        if verbosity >= 1
            println("\tnew_active: $(new_active)")
            println("\tactive predictors: ", n_active_i)
        end

        if screening_type == "hessian_adaptive"
            updateGradientOfCorrelation!(
                family,
                ∇c,
                c,
                residual,
                y,
                s,
                X,
                X_mean_scaled,
                active_set,
                inactive_set,
                1:p,
                H⁻s,
                standardize
            )

            λ_next, λ_next_mod = getNextLambda(
                c,
                ∇c,
                λ,
                λ_prev,
                λ_min,
                λ_minstep,
                λ_next_mod,
                inactive_set,
                n_target_nonzero,
                new_active,
                i,
                verbosity
            )
        else
            λ_next = λs[i + 1]
        end

        # compute strong set
        strong = abs.(c) .>= (2*λ_next - λ)
        push!(n_strong, sum(strong))

        t0 = time()

        if screening_type == "hessian"
            updateGradientOfCorrelation!(family,
                                         ∇c,
                                         c,
                                         residual,
                                         y,
                                         s,
                                         X,
                                         X_mean_scaled,
                                         active_set,
                                         inactive_set,
                                         findall(strong),
                                         H⁻s,
                                         standardize)
        end

        push!(gradcorr_time, time() - t0)

        screenPredictors!(screened,
                          strong,
                          c,
                          ∇c,
                          X,
                          X_mean_scaled,
                          X²sums,
                          y,
                          residual,
                          dual_scale,
                          λ,
                          λ_next,
                          γ,
                          ever_active,
                          duplicates,
                          screening_type,
                          standardize)

        # warm start next beta based on hessian solution
        if hessian_warm_starts && screening_type in ["hessian", "hessian_adaptive"]
            β[active_set] = β[active_set] + (λ - λ_next)*H⁻s
        end

        λ = λ_next

        active_set_prev = copy(active_set)

        if screening_type == "hessian_adaptive"
            push!(λs, λ)
        end
    end

    full_time = time() - full_time
    path_time = cd_time_list .+ hessian_time_list

    (β = βs,
     γ = γs,
     λ = λs[1:i],
     primals = primals,
     duals = duals,
     dev_ratios = dev_ratios,
     dev = devs,
     refits = refits,
     duplicate_matrix = [originals duplicates],
     violations = violations_list,
     n_active = n_active,
     n_screened = n_screened,
     n_new_active,
     n_strong = n_strong,
     passes = passes,
     path_time = path_time,
     cd_time =  cd_time_list,
     gradcorr_time = gradcorr_time,
     hessian_time = hessian_time_list,
     corr_time = corr_time_list,
     full_time = full_time)
end

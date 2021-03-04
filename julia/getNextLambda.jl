# find lambda at approximately the next point at which some n_target_nonzero predictors
# enter the model
function getNextLambda(c,
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
                       verbosity)

    if isempty(inactive_set)
        return λ_min, λ_next_mod
    end

    if i > 1
        # adjust target nonzero to avoid stepping too far at end of path
        if new_active > n_target_nonzero
            λ_next_mod *= 1.01
        elseif new_active < n_target_nonzero
            λ_next_mod /= 1.01
        end

        n_target_nonzero = min(n_target_nonzero, length(inactive_set))

        if verbosity >= 1
            println("\tλ_next_mod: $(λ_next_mod)")
        end
    end

    a = (c[inactive_set] .- λ .* ∇c[inactive_set]) ./ (1.0 .- ∇c[inactive_set])
    b = (λ .* ∇c[inactive_set] .- c[inactive_set]) ./ (1.0 .+ ∇c[inactive_set])

    a[a .>= λ] .= 0.0
    b[b .>= λ] .= 0.0

    # pick lambda predicting that approximately n_target_nonzero
    # predictors enter the model at the next step
    λ_next = sort(max.(a, b), rev = true)[n_target_nonzero]

    # adjust next λ slightly based on compliance with
    # intended number of new nonzero predictors per step
    λ_next *= λ_next_mod

    λ_change = λ - λ_next

    if λ_change < λ_minstep || λ_change < 0
        λ_next = max(λ_prev - λ_minstep, λ_min)
    end

    λ_next, λ_next_mod
end
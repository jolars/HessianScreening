include("kktCheck.jl")
include("gaussian.jl")

# check for violations and update screened set where they occur
function checkViolations(X,
                         X_mean_scaled,
                         residual,
                         residual_sum,
                         c,
                         λ,
                         screened,
                         strong,
                         duplicates,
                         screening_type;
                         standardize = true)

    violations = falses(size(X, 2))

    unscreened = .!screened

    if screening_type == "strong"
        kktCheck!(violations, c, λ, screened, unscreened)
    else
        # check for violations in strong set
        check_set = findall(strong .& unscreened)
        check_set = setdiff(check_set, duplicates)
        # println("small checkset = $(length(check_set))")

        updateCorrelation!(c, X, X_mean_scaled, residual, check_set, standardize)
        kktCheck!(violations, c, λ, screened, check_set)

        #violations[unscreened] .= false
        #screened[violations] .= false

        #violations[duplicates] .= false

        if !any(violations)
            # check for violations in remaining predictors
            check_set = findall(.!strong .& unscreened)
            check_set = setdiff(check_set, duplicates)

            # println("big checkset = $(length(check_set))")
            updateCorrelation!(c, X, X_mean_scaled, residual, check_set, standardize)
            kktCheck!(violations, c, λ, screened, check_set)
        end
    end

    sum(violations)
end

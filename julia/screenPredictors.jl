
function screenPredictors!(screened,
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

    if screening_type == "working"
        screened[1:end] = ever_active
    elseif screening_type == "strong"
        screened[1:end] = strong
    elseif screening_type in ["hessian", "hessian_adaptive"]
        c_pred = c .+ ∇c * (λ_next - λ)
        screened[1:end] = (abs.(c_pred) .+ γ*(λ - λ_next)) .> λ_next
        screened[1:end] = screened .| ever_active

        # ensure that duplicates stay out
        screened[duplicates] .= false
    elseif screening_type in ["gap_safe"]
        # we use the active set strategy for the GAP-safe rules, so start
        # with a CD loop over the ever-active predictors as a warm start
        screened .= ever_active
    elseif screening_type in ["edpp"]
        n = length(y)
        p = length(c)

        v1 = y ./ λ .- residual ./ dual_scale
        v2 = y ./ λ_next .- residual ./ dual_scale

        norm_v1 = norm(v1)^2
        v_orth = norm_v1 != 0 ? v2 .- (dot(v1, v2) / norm_v1) .* v1 : copy(v2)
        center = residual ./ dual_scale .+ 0.5 * v_orth
        r_screen = 0.5 * norm(v_orth)

        XTcenter = zeros(p)

        for j in 1:p
            XTcenter[j] = dot(X[:, j], center)
        end

        screened .= r_screen .* sqrt.(X²sums) + abs.(XTcenter) .>= 1.0
    end
end
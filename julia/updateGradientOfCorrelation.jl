function updateGradientOfCorrelation!(::Normal,
                                      ∇c,
                                      c,
                                      residual,
                                      y,
                                      s,
                                      X,
                                      X_mean_scaled,
                                      active_set,
                                      inactive,
                                      restricted,
                                      H⁻s,
                                      standardize)

    inactive_restricted = intersect(inactive, restricted)
    inactive_notrestricted = setdiff(inactive, restricted)

    if issparse(X) && standardize
        tmp = X[:, active_set]*H⁻s .- dot(X_mean_scaled[active_set], H⁻s)
        ∇c[inactive_restricted] =
            X[:, inactive_restricted]'tmp - X_mean_scaled[inactive_restricted]*sum(tmp)
    else
        ∇c[inactive_restricted] = X[:, inactive_restricted]'*(X[:, active_set] * H⁻s)
    end

    ∇c[inactive_notrestricted] .= 0.0
    ∇c[active_set] = s[active_set]
end

function updateGradientOfCorrelation!(::Binomial,
                                      ∇c,
                                      c,
                                      residual,
                                      y,
                                      s,
                                      X,
                                      X_mean_scaled,
                                      active_set,
                                      inactive,
                                      restricted,
                                      H⁻s,
                                      standardize)

    inactive_restricted = intersect(inactive, restricted)
    inactive_notrestricted = setdiff(inactive, restricted)

    p = clamp.(y .- residual, P_MIN, P_MAX)
    # ϵ = 1e-12
    # p_hi = p .> (1 - ϵ)
    # p_lo = p .< ϵ

    w = p .* (1.0 .- p)
    # w[p_hi] .= ϵ
    # w[p_lo] .= 1 - ϵ

    dsq = sqrt.(w)
    D = Diagonal(w)
    Dsq = Diagonal(dsq)

    if issparse(X) && standardize
        DsqX = Dsq * X[:, inactive_restricted]
        DsqXa = Dsq * X[:, active_set]

        dsqmu = dsq * X_mean_scaled[inactive_restricted]'
        dsqmuaH⁻s = (dsq * (X_mean_scaled[active_set]' * H⁻s))
        DsqXaH⁻s = DsqXa * H⁻s

        tmp = DsqX' * (DsqXaH⁻s - dsqmuaH⁻s) + dsqmu'*(dsqmuaH⁻s - DsqXaH⁻s)

        ∇c[inactive_restricted] = tmp
    else
        ∇c[inactive_restricted] = X[:, inactive_restricted]'*D*(X[:, active_set] * H⁻s)
    end

    ∇c[inactive_notrestricted] .= 0.0
    ∇c[active_set] = s[active_set]
end

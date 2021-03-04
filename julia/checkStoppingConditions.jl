function checkStoppingConditions(i,
                                 n,
                                 p,
                                 n_λ,
                                 n_active,
                                 λ,
                                 λ_min,
                                 dev,
                                 dev_prev,
                                 null_dev,
                                 screening_type,
                                 verbosity)
    if verbosity >= 1
        println("\tchecking stopping conditions")
    end

    if i == 1
        # first step, always continue
        return false
    end

    dev_ratio  = 1.0 - dev/null_dev
    dev_change = 1.0 - dev/dev_prev

    if verbosity >= 1
        println("\t\tdev change: ", dev_change)
        println("\t\tdev ratio: ", dev_ratio)
    end

    if dev_change <= 1e-5
        return true
    end

    if dev_ratio >= 0.999 || λ <= λ_min
        return true
    end

    if (n <= p) && (n_active >= n)
        return true
    end

    if screening_type != "hessian_adaptive" && i >= n_λ
        return true
    end

    false
end

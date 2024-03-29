# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

lassoPathDense <- function(X, y, family, lambdas, lambda_type, standardize, screening_type, shuffle, check_frequency, screen_frequency, hessian_warm_starts, celer_use_old_dual, celer_use_accel, celer_prune, gap_safe_active_start, augment_with_gap_safe, update_hessian, log_hessian_update_type, log_hessian_auto_update_freq, path_length, maxit, tol_gap, gamma, store_dual_variables, verify_hessian, line_search, verbosity) {
    .Call(`_HessianScreening_lassoPathDense`, X, y, family, lambdas, lambda_type, standardize, screening_type, shuffle, check_frequency, screen_frequency, hessian_warm_starts, celer_use_old_dual, celer_use_accel, celer_prune, gap_safe_active_start, augment_with_gap_safe, update_hessian, log_hessian_update_type, log_hessian_auto_update_freq, path_length, maxit, tol_gap, gamma, store_dual_variables, verify_hessian, line_search, verbosity)
}

lassoPathSparse <- function(X, y, family, lambdas, lambda_type, standardize, screening_type, shuffle, check_frequency, screen_frequency, hessian_warm_starts, celer_use_old_dual, celer_use_accel, celer_prune, gap_safe_active_start, augment_with_gap_safe, update_hessian, log_hessian_update_type, log_hessian_auto_update_freq, path_length, maxit, tol_gap, gamma, store_dual_variables, verify_hessian, line_search, verbosity) {
    .Call(`_HessianScreening_lassoPathSparse`, X, y, family, lambdas, lambda_type, standardize, screening_type, shuffle, check_frequency, screen_frequency, hessian_warm_starts, celer_use_old_dual, celer_use_accel, celer_prune, gap_safe_active_start, augment_with_gap_safe, update_hessian, log_hessian_update_type, log_hessian_auto_update_freq, path_length, maxit, tol_gap, gamma, store_dual_variables, verify_hessian, line_search, verbosity)
}


#' Lasso Path with Hessian Screening Rules
#'
#' @param X The predictor matrix
#' @param y The reponse vector
#' @param family The name of the family, "gaussian" or "logistic"
#' @param standardize Whether to standardize the predictors
#' @param screening_type Screening rule
#' @param hessian_warm_starts Whether to use warm starts based on Hessian
#' @param hessian_warm_starts Whether to use the active start strategy for
#'   the Gap-Safe rule
#' @param log_hessian_update_type What type of strategy to use for
#'   updating the hessian for logistic regression
#' @param log_hessian_auto_update_freq Frequency of hessian updates when
#'   `log_hessian_update_type = "auto"`
#' @param path_length The (desired) length of the lasso path
#' @param maxit Maximum number of iterations for Coordinate Descent loop
#' @param tol_gap Tolerance threshold for duality gap
#' @param gamma Percent of strong approximation to add to Hessian approximation
#' @param verify_hessian Whether to not to verify that Hessian updates are
#'   correct. Used only for diagnostic purposes.
#' @param force_kkt_check Whether to force KKT checks even when screening rule
#'   is safe
#' @param line_search Use line search when `family = "binomial"`
#' @param verbosity Controls the level of verbosity. 0 = no output.
#'
#' @export
lassoPath <- function(X,
                      y,
                      family = c("gaussian", "binomial"),
                      standardize = TRUE,
                      screening_type = c(
                        "working",
                        "hessian",
                        "hessian_adaptive",
                        "edpp",
                        "gap_safe",
                        "strong",
                        "celer"
                      ),
                      hessian_warm_starts = TRUE,
                      celer_use_old_dual = TRUE,
                      celer_use_accel = TRUE,
                      celer_prune = FALSE,
                      gap_safe_active_start = TRUE,
                      log_hessian_update_type = c("full", "auto", "approx"),
                      log_hessian_auto_update_freq = 10,
                      path_length = 100L,
                      maxit = 1e5,
                      tol_gap = 1e-5,
                      gamma = 0.01,
                      verify_hessian = FALSE,
                      force_kkt_check = FALSE,
                      line_search = 3,
                      verbosity = 0) {
  family <- match.arg(family)
  screening_type <- match.arg(screening_type)
  log_hessian_update_type <- match.arg(log_hessian_update_type)

  sparse <- inherits(X, "sparseMatrix")

  n <- nrow(X)
  p <- ncol(X)

  if (sparse) {
    X <- as(X, "dgCMatrix")

    lassoPathSparse(
      X,
      y,
      family,
      standardize,
      screening_type,
      hessian_warm_starts,
      celer_use_old_dual,
      celer_use_accel,
      celer_prune,
      gap_safe_active_start,
      log_hessian_update_type,
      log_hessian_auto_update_freq,
      path_length,
      maxit,
      tol_gap,
      gamma,
      verify_hessian,
      force_kkt_check,
      line_search,
      verbosity
    )
  } else {
    lassoPathDense(
      X,
      y,
      family,
      standardize,
      screening_type,
      hessian_warm_starts,
      celer_use_old_dual,
      celer_use_accel,
      celer_prune,
      gap_safe_active_start,
      log_hessian_update_type,
      log_hessian_auto_update_freq,
      path_length,
      maxit,
      tol_gap,
      gamma,
      verify_hessian,
      force_kkt_check,
      line_search,
      verbosity
    )
  }
}

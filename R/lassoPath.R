#' Lasso Path with Hessian Screening Rules
#'
#' This function fits the full lasso path.
#'
#' @param X The predictor matrix
#' @param y The reponse vector
#' @param family The name of the family, "gaussian" or "logistic"
#' @param standardize Whether to standardize the predictors
#' @param screening_type Screening rule
#' @param shuffle Shuffle working set before each CD pass?
#' @param check_frequency Frequency at which duality gap is checked for
#'   inner CD loop
#' @param screen_frequency Frequency at which predictors are screened for
#'   the gap safe solver
#' @param hessian_warm_starts Whether to use warm starts based on Hessian
#' @param gap_safe_active_start Whether to use the active start strategy for
#'   the Gap-Safe rule
#' @param augment_with_gap_safe Whether or not augment heuristic rules
#'   with Gap Safe checks during KKT checks
#' @param log_hessian_update_type What type of strategy to use for
#'   updating the hessian for logistic regression
#' @param log_hessian_auto_update_freq Frequency of hessian updates when
#'   `log_hessian_update_type = "auto"`
#' @param path_length The (desired) length of the lasso path
#' @param maxit Maximum number of iterations for Coordinate Descent loop
#' @param tol_gap Tolerance threshold for relative duality gap.
#' @param gamma Percent of strong approximation to add to Hessian approximation
#' @param verify_hessian Whether to not to verify that Hessian updates are
#'   correct. Used only for diagnostic purposes.
#' @param line_search Use line search in CD solver.
#' @param verbosity Controls the level of verbosity. 0 = no output, 1 = outer
#'   level output, 2 = inner solver output
#' @param lambda weights for the regularization path, if `NULL`, then they
#'   are automatically computed
#' @param celer_use_old_dual whether to try to use previous dual when checking
#'   duality gap globally and screening for Celer
#' @param celer_use_accel whether to use accelerated dual point for Celer
#' @param celer_prune whether to use pruning for Celer
#' @param store_dual_variables whether to store dual variables throughout
#'   fitting
#'
#' @export
lassoPath <- function(X,
                      y,
                      family = c("gaussian", "binomial"),
                      lambda = NULL,
                      standardize = TRUE,
                      screening_type = c(
                        "hessian",
                        "working",
                        "edpp",
                        "gap_safe",
                        "strong",
                        "celer",
                        "blitz",
                        "sasvi"
                      ),
                      shuffle = match.arg(screening_type) == "blitz",
                      check_frequency = if (NROW(X) > NCOL(X)) 1 else 10,
                      screen_frequency = 10,
                      hessian_warm_starts = TRUE,
                      celer_use_old_dual = TRUE,
                      celer_use_accel = TRUE,
                      celer_prune = TRUE,
                      gap_safe_active_start = TRUE,
                      augment_with_gap_safe = TRUE,
                      log_hessian_update_type = c("full", "auto", "approx"),
                      log_hessian_auto_update_freq = 10,
                      path_length = 100L,
                      maxit = 1e5,
                      tol_gap = 1e-4,
                      gamma = 0.01,
                      store_dual_variables = FALSE,
                      verify_hessian = FALSE,
                      line_search = TRUE,
                      verbosity = 0) {
  n <- nrow(X)
  p <- ncol(X)

  family <- match.arg(family)
  screening_type <- match.arg(screening_type)
  log_hessian_update_type <- match.arg(log_hessian_update_type)

  if (is.null(lambda)) {
    lambda <- double(path_length)
    lambda_type <- "auto"
  } else {
    lambda_type <- "user"
  }

  sparse <- inherits(X, "sparseMatrix")

  if (sparse) {
    X <- as(X, "dgCMatrix")

    lassoPathSparse(
      X,
      y,
      family,
      lambda,
      lambda_type,
      standardize,
      screening_type,
      shuffle,
      check_frequency,
      screen_frequency,
      hessian_warm_starts,
      celer_use_old_dual,
      celer_use_accel,
      celer_prune,
      gap_safe_active_start,
      augment_with_gap_safe,
      log_hessian_update_type,
      log_hessian_auto_update_freq,
      path_length,
      maxit,
      tol_gap,
      gamma,
      store_dual_variables,
      verify_hessian,
      line_search,
      verbosity
    )
  } else {
    lassoPathDense(
      X,
      y,
      family,
      lambda,
      lambda_type,
      standardize,
      screening_type,
      shuffle,
      check_frequency,
      screen_frequency,
      hessian_warm_starts,
      celer_use_old_dual,
      celer_use_accel,
      celer_prune,
      gap_safe_active_start,
      augment_with_gap_safe,
      log_hessian_update_type,
      log_hessian_auto_update_freq,
      path_length,
      maxit,
      tol_gap,
      gamma,
      store_dual_variables,
      verify_hessian,
      line_search,
      verbosity
    )
  }
}

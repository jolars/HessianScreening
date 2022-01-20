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
#' @param hessian_warm_starts Whether to use the active start strategy for
#'   the Gap-Safe rule
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
                        "celer",
                        "blitz"
                      ),
                      shuffle = match.arg(screening_type) == "blitz",
                      check_frequency = if (NROW(X) > NCOL(X)) 1 else 10,
                      screen_frequency = 10,
                      hessian_warm_starts = TRUE,
                      celer_use_old_dual = TRUE,
                      celer_use_accel = TRUE,
                      celer_prune = FALSE,
                      gap_safe_active_start = TRUE,
                      log_hessian_update_type = c("full", "auto", "approx"),
                      log_hessian_auto_update_freq = 10,
                      path_length = 100L,
                      maxit = 1e5,
                      tol_gap = 1e-4,
                      gamma = 0.01,
                      store_dual_variables = FALSE,
                      verify_hessian = FALSE,
                      line_search = NULL,
                      verbosity = 0) {
  n <- nrow(X)
  p <- ncol(X)

  family <- match.arg(family)
  screening_type <- match.arg(screening_type)
  log_hessian_update_type <- match.arg(log_hessian_update_type)

  if (is.null(line_search)) {
    if (screening_type == "blitz") {
      line_search <- 1
    } else {
      line_search <- 0
    }
  }

  stopifnot(line_search %in% c(0, 1, 2))

  sparse <- inherits(X, "sparseMatrix")

  if (sparse) {
    X <- as(X, "dgCMatrix")

    lassoPathSparse(
      X,
      y,
      family,
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

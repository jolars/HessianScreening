renderPdf <- function(x) {
  wd <- getwd()
  on.exit({
    setwd(wd)
  })

  path <- normalizePath(dirname(x))

  full_file_path <- tools::file_path_as_absolute(x)
  file_wo_ext <- tools::file_path_sans_ext(basename(x))

  pdf_file <- paste0(file_wo_ext, ".pdf")

  # work in a temporary directory to avoid dealing with latex log files
  tmp_dir <- tempdir()
  setwd(tmp_dir)

  tools::texi2pdf(full_file_path)
  knitr:::plot_crop(pdf_file)
  file.copy(pdf_file, file.path(path, pdf_file), overwrite = TRUE)
}

gaussian_primal <- function(x, y, beta, lambda) {
  0.5 * norm(y - x %*% beta, "2")^2 + lambda * sum(abs(beta))
}

gaussian_dual <- function(x, y, beta, lambda) {
  residual <- y - x %*% beta
  correlation <- Matrix::crossprod(x, residual)

  theta <- residual / max(lambda, max(abs(correlation)))

  0.5 * norm(y, "2")^2 - 0.5 * lambda^2 * norm(theta - y / lambda, "2")^2
}

binomial_primal <- function(x, y, beta, lambda) {
  xbeta <- x %*% beta
  -sum(y * xbeta - log1p(exp(xbeta))) + lambda * sum(abs(beta))
}

binomial_dual <- function(x, y, beta, lambda) {
  exp_xbeta <- exp(x %*% beta)
  pr <- exp_xbeta / (1 + exp_xbeta)
  pr <- ifelse(pr < 1e-5, 1e-5, pr)
  pr <- ifelse(pr > 1 - 1e-5, 1 - 1e-5, pr)

  residual <- y - pr

  correlation <- Matrix::crossprod(x, residual)

  theta <- residual / max(lambda, max(abs(correlation)))

  prx <- y - lambda * theta
  prx <- ifelse(prx < 1e-5, 1e-5, prx)
  prx <- ifelse(prx > 1 - 1e-5, 1 - 1e-5, prx)

  -sum(prx * log(prx) + (1 - prx) * log(1 - prx))
}

#' Get Duality Gaps
#'
#' @param fit the resulting fit
#' @param family the loss function
#' @param x design matrix
#' @param y response vector
#'
#' @return primals, duals, and relative duality gaps
#' @export
check_gaps <- function(fit, family, standardize, x, y, tol_gap = 1e-4) {
  beta <- fit$beta
  lambda <- fit$lambda

  duals <- primals <- double(length(lambda))

  n <- length(y)

  for (i in seq_along(duals)) {
    if (family == "gaussian") {
      primals[i] <- gaussian_primal(x, y, beta[, i], lambda[i])
      duals[i] <- gaussian_dual(x, y, beta[, i], lambda[i])
    } else if (family == "binomial") {
      primals[i] <- binomial_primal(x, y, beta[, i], lambda[i])
      duals[i] <- binomial_dual(x, y, beta[, i], lambda[i])
    }
  }

  tol_gap_rel <- if (family == "gaussian") {
    tol_gap * norm(y, "2")^2 
  } else if (family == "binomial") {
    tol_gap * n * log(2)
  }

  list(
    primals = primals,
    duals = duals,
    gaps = primals - duals,
    tol = tol_gap_rel,
    below_tol = (primals - duals) <= tol_gap_rel
  )
}

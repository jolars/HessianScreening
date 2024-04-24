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

gaussian_dual <- function(y, theta, lambda) {
  0.5 * norm(y, "2")^2 - 0.5 * lambda^2 * norm(theta - y / lambda, "2")^2
}

binomial_primal <- function(x, y, beta, lambda) {
  xbeta <- x %*% beta
  -sum(y * xbeta - log1p(exp(xbeta))) + lambda * sum(abs(beta))
}

binomial_dual <- function(y, theta, lambda) {
#   exp_xbeta <- exp(x %*% beta)
#   pr <- exp_xbeta / (1 + exp_xbeta)
#   pr <- ifelse(pr < 1e-5, 1e-5, pr)
#   pr <- ifelse(pr > 1 - 1e-5, 1 - 1e-5, pr)

#   residual <- y - pr

#   correlation <- Matrix::crossprod(x, residual)

#   theta <- residual / max(lambda, max(abs(correlation)))

  prx <- y - lambda * theta
  prx <- ifelse(prx < 1e-5, 1e-5, prx)
  prx <- ifelse(prx > 1 - 1e-5, 1 - 1e-5, prx)

  -sum(prx * log(prx) + (1 - prx) * log(1 - prx))
}

#' Get Duality Gaps
#'
#' @param fit the resulting fit
#' @param x design matrix
#' @param y response vector
#' @param standardize whether the fit used standardization
#' @param tol_gap the tolerance threshold used when fitting
#'
#' @return primals, duals, and relative duality gaps
#' @export
check_gaps <- function(fit, standardize, x, y, tol_gap = 1e-4) {
  beta <- fit$beta
  theta <- fit$theta
  lambda <- fit$lambda
  family <- fit$family

  if (ncol(theta) == 0) {
    stop("please call fitLasso() with `store_dual_variables = TRUE`")
  }

  if (standardize) {
    pop_sd <- function(a) sqrt((length(a) - 1) / length(a)) * sd(a)

    s <- apply(x, 2, pop_sd)
    beta <- beta * s

    x <- scale(x, scale = s)
  }

  if (family == "gaussian")
    y <- y - mean(y)

  duals <- primals <- double(length(lambda))

  n <- length(y)

  for (i in seq_along(duals)) {
    if (family == "gaussian") {
      primals[i] <- gaussian_primal(x, y, beta[, i], lambda[i])
      duals[i] <- gaussian_dual(y, theta[, i], lambda[i])
    } else if (family == "binomial") {
      primals[i] <- binomial_primal(x, y, beta[, i], lambda[i])
      duals[i] <- binomial_dual(y, theta[, i], lambda[i])
    }
  }

  tol_mod <- if (family == "gaussian") norm(y, "2")^2 else log(2)*length(y)

  list(
    primals = primals,
    duals = duals,
    gaps = primals - duals,
    rel_gaps = (primals - duals) / tol_mod,
    below_tol = primals - duals <= tol_gap * tol_mod
  )
}

recode_methods <- function(x) {
  screening_type = factor(
    x,
    levels = c(
      "hessian",
      "working",
      "celer",
      "blitz",
      "strong",
      "edpp",
      "gap_safe",
      "sasvi"
    ),
    labels = c(
      "Hessian",
      "Working",
      "Celer",
      "Blitz",
      "Strong",
      "EDPP",
      "Gap Safe",
      "Sasvi"
    )
  )
}

rho_labeller <- function(labels, multi_line = TRUE, sep = ":", ...) {
  value <- label_value(labels, multi_line = multi_line)
  out <- paste0("$\\rho = ", value, "$")
  #  list(unname(unlist(out)))
  out
}

color_palette <- function(name = c("wong", "tableau"), n = NULL) {
  name <- match.arg(name)

  wong <- c(
    "#999999",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7"
  )
  
  tableau <- c(
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac"
  ) 

  pal <- switch(name, wong = wong, tableau = tableau)

  if (is.null(n)) {
    n <- length(pal)
  }

  if (n > length(pal)) {
    stop("longer length than pal")
  }

  pal[seq_len(n)]
}

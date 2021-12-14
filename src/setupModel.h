#pragma once

#include "model.h"
#include <RcppArmadillo.h>

std::unique_ptr<Model>
setupModel(const std::string family,
           arma::vec& y,
           arma::vec& beta,
           arma::vec& residual,
           arma::vec& Xbeta,
           arma::vec& c,
           const arma::vec& X_mean_scaled,
           const arma::vec& X_norms_squared,
           const arma::uword n,
           const arma::uword p,
           const bool standardize,
           const std::string log_hessian_update_type);

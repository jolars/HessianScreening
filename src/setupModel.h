#pragma once

#include <RcppArmadillo.h>

#include "model.h"
#include "binomial.h"
#include "gaussian.h"

using namespace arma;

std::unique_ptr<Model>
setupModel(const std::string family,
           vec& y,
           vec& beta,
           vec& residual,
           vec& Xbeta,
           vec& c,
           const vec& X_mean_scaled,
           const vec& X_norms_squared,
           const uword n,
           const uword p,
           const bool standardize,
           const std::string log_hessian_update_type)
{
  if (family == "binomial")
    return std::make_unique<Binomial>(y,
                                      beta,
                                      residual,
                                      Xbeta,
                                      c,
                                      X_mean_scaled,
                                      X_norms_squared,
                                      n,
                                      p,
                                      standardize,
                                      log_hessian_update_type);

  return std::make_unique<Gaussian>(y,
                                    beta,
                                    residual,
                                    Xbeta,
                                    c,
                                    X_mean_scaled,
                                    X_norms_squared,
                                    n,
                                    p,
                                    standardize);
}
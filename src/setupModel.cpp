#include "binomial.h"
#include "gaussian.h"
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
           const std::string log_hessian_update_type)
{
  if (family == "binomial")
    return std::make_unique<Binomial>(family,
                                      y,
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

  return std::make_unique<Gaussian>(family,
                                    y,
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

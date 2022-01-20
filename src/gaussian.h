#pragma once

#include "model.h"
#include <RcppArmadillo.h>

class Gaussian : public Model
{
public:
  const arma::vec& X_norms_squared;

  Gaussian(const std::string family, const arma::vec& X_norms_squared);

  double primal(const arma::vec& residual,
                const arma::vec& Xbeta,
                const arma::vec& beta,
                const arma::vec& y,
                const double lambda);

  double primal(const arma::vec& residual,
                const arma::vec& Xbeta,
                const arma::vec& beta,
                const arma::vec& y,
                const double lambda,
                const arma::uvec& screened_set);

  double dual(const arma::vec& theta, const arma::vec& y, const double lambda);

  double deviance(const arma::vec& residual,
                  const arma::vec& Xbeta,
                  const arma::vec& y);

  void updateResidual(arma::vec& residual,
                      const arma::vec& Xbeta,
                      const arma::vec& y);

  void adjustResidual(arma::vec& residual,
                      arma::vec& Xbeta,
                      const arma::mat& X,
                      const arma::vec& y,
                      const arma::uword j,
                      const double beta_diff,
                      const arma::vec& X_offset,
                      const bool standardize);

  void adjustResidual(arma::vec& residual,
                      arma::vec& Xbeta,
                      const arma::sp_mat& X,
                      const arma::vec& y,
                      const arma::uword j,
                      const double beta_diff,
                      const arma::vec& X_offset,
                      const bool standardize);

  arma::vec weights(const arma::vec& residual, const arma::vec& y);

  arma::mat hessian(const arma::mat& X,
                    const arma::uvec& ind,
                    const arma::vec& X_offset,
                    const bool standardize);

  arma::mat hessian(const arma::sp_mat& X,
                    const arma::uvec& ind,
                    const arma::vec& X_offset,
                    const bool standardize);

  arma::mat hessianUpperRight(const arma::mat& X,
                              const arma::uvec& ind_a,
                              const arma::uvec& ind_b,
                              const arma::vec& X_offset,
                              const bool standardize);

  arma::mat hessianUpperRight(const arma::sp_mat& X,
                              const arma::uvec& ind_a,
                              const arma::uvec& ind_b,
                              const arma::vec& X_offset,
                              const bool standardize);

  double hessianTerm(const arma::mat& X,
                     const arma::uword j,
                     const arma::vec& X_offset,
                     const bool standardize);

  double hessianTerm(const arma::sp_mat& X,
                     const arma::uword j,
                     const arma::vec& X_offset,
                     const bool standardize);

  void updateGradientOfCorrelation(arma::vec& c_grad,
                                   const arma::mat& X,
                                   const arma::vec& Hinv_s,
                                   const arma::vec& s,
                                   const arma::uvec& active_set,
                                   const arma::uvec& restricted_set,
                                   const arma::vec& X_offset,
                                   const bool standardize);

  void updateGradientOfCorrelation(arma::vec& c_grad,
                                   const arma::sp_mat& X,
                                   const arma::vec& Hinv_s,
                                   const arma::vec& s,
                                   const arma::uvec& active_set,
                                   const arma::uvec& restricted_set,
                                   const arma::vec& X_offset,
                                   const bool standardize);

  void standardizeY(arma::vec& y);

  double safeScreeningRadius(const double duality_gap, const double lambda);

  double toleranceModifier(const arma::vec& y);
};

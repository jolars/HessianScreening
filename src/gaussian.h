#pragma once

#include "model.h"
#include "prox.hpp"
#include "utils.hpp"
#include <RcppArmadillo.h>

class Gaussian : public Model
{
public:
  Gaussian(const std::string family,
           arma::vec& y,
           arma::vec& beta,
           arma::vec& residual,
           arma::vec& Xbeta,
           arma::vec& c,
           const arma::vec& X_mean_scaled,
           const arma::vec& X_norms_squared,
           const arma::uword n,
           const arma::uword p,
           const bool standardize);

  double primal(const double lambda, const arma::uvec& screened_set);

  double dual();

  double dual(const double lambda, const arma::vec& theta);

  double scaledDual(const double lambda, const double dual_scale);

  double deviance();

  double hessianTerm(const arma::mat& X, const arma::uword j);

  double hessianTerm(const arma::sp_mat& X, const arma::uword j);

  void updateResidual();

  void adjustResidual(const arma::mat& X,
                      const arma::uword j,
                      const double beta_diff);

  void adjustResidual(const arma::sp_mat& X,
                      const arma::uword j,
                      const double beta_diff);

  arma::mat hessian(const arma::mat& X, const arma::uvec& ind);

  arma::mat hessian(const arma::sp_mat& X, const arma::uvec& ind);

  arma::mat hessianUpperRight(const arma::mat& X,
                              const arma::uvec& ind_a,
                              const arma::uvec& ind_b);

  arma::mat hessianUpperRight(const arma::sp_mat& X,
                              const arma::uvec& ind_a,
                              const arma::uvec& ind_b);

  void updateGradientOfCorrelation(arma::vec& c_grad,
                                   const arma::mat& X,
                                   const arma::vec& Hinv_s,
                                   const arma::vec& s,
                                   const arma::uvec& active_set,
                                   const arma::uvec& restricted_set);

  void updateGradientOfCorrelation(arma::vec& c_grad,
                                   const arma::sp_mat& X,
                                   const arma::vec& Hinv_s,
                                   const arma::vec& s,
                                   const arma::uvec& active_set,
                                   const arma::uvec& restricted_set);

  void standardizeY();

  double safeScreeningRadius(const double duality_gap, const double lambda);
};

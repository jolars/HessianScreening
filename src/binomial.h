#pragma once

#include "model.h"
#include <RcppArmadillo.h>

class Binomial : public Model
{
public:
  arma::vec expXbeta;
  arma::vec pr;
  arma::vec w;

  std::string log_hessian_update_type;

  const double p_min = 1e-5;
  const double p_max = 1 - p_min;

  Binomial(const std::string family,
           arma::vec& y,
           arma::vec& beta,
           arma::vec& Xbeta,
           const arma::vec& X_mean_scaled,
           const arma::vec& X_norms_squared,
           const arma::uword n,
           const arma::uword p,
           const bool standardize,
           const std::string log_hessian_update_type);

  void setLogHessianUpdateType(const std::string new_log_hessian_update_type);

  double primal(const arma::vec& residual, const double lambda);

  double primal(const arma::vec& residual,
                const double lambda,
                const arma::uvec& screened_set);

  double dual(const arma::vec& theta, const arma::vec& y, const double lambda);

  double deviance(const arma::vec& residual);

  void updateResidual(arma::vec& residual);

  void adjustResidual(arma::vec& residual,
                      const arma::mat& X,
                      const arma::uword j,
                      const double beta_diff);

  void adjustResidual(arma::vec& residual,
                      const arma::sp_mat& X,
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

  double hessianTerm(const arma::mat& X, const arma::uword j);

  double hessianTerm(const arma::sp_mat& X, const arma::uword j);


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

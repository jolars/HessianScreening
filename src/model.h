#pragma once

#include "prox.h"
#include <RcppArmadillo.h>

class Model
{
public:
  const std::string family;

  Model(const std::string family);

  virtual ~Model() = default;

  void setLogHessianUpdateType(const std::string new_log_hessian_update_type);

  virtual double primal(const arma::vec& residual,
                        const arma::vec& Xbeta,
                        const arma::vec& beta,
                        const arma::vec& y,
                        const double lambda) = 0;

  virtual double primal(const arma::vec& residual,
                        const arma::vec& Xbeta,
                        const arma::vec& beta,
                        const arma::vec& y,
                        const double lambda,
                        const arma::uvec& screened_set) = 0;

  virtual double dual(const arma::vec& theta,
                      const arma::vec& y,
                      const double lambda) = 0;

  virtual double deviance(const arma::vec& residual,
                          const arma::vec& Xbeta,
                          const arma::vec& y) = 0;

  virtual void updateResidual(arma::vec& residual,
                              const arma::vec& Xbeta,
                              const arma::vec& y) = 0;

  virtual void adjustResidual(arma::vec& residual,
                              arma::vec& Xbeta,
                              const arma::mat& X,
                              const arma::vec& y,
                              const arma::uword j,
                              const double beta_diff,
                              const arma::vec& X_offset,
                              const bool standardize) = 0;

  virtual void adjustResidual(arma::vec& residual,
                              arma::vec& Xbeta,
                              const arma::sp_mat& X,
                              const arma::vec& y,
                              const arma::uword j,
                              const double beta_diff,
                              const arma::vec& X_offset,
                              const bool standardize) = 0;

  virtual arma::vec weights(const arma::vec& residual, const arma::vec& y) = 0;

  virtual arma::mat hessian(const arma::mat& X,
                            const arma::uvec& ind,
                            const arma::vec& X_offset,
                            const bool standardize) = 0;

  virtual arma::mat hessian(const arma::sp_mat& X,
                            const arma::uvec& ind,
                            const arma::vec& X_offset,
                            const bool standardize) = 0;

  virtual arma::mat hessianUpperRight(const arma::mat& X,
                                      const arma::uvec& ind_a,
                                      const arma::uvec& ind_b,
                                      const arma::vec& X_offset,
                                      const bool standardize) = 0;

  virtual arma::mat hessianUpperRight(const arma::sp_mat& X,
                                      const arma::uvec& ind_a,
                                      const arma::uvec& ind_b,
                                      const arma::vec& X_offset,
                                      const bool standardize) = 0;

  virtual double hessianTerm(const arma::mat& X,
                             const arma::uword j,
                             const arma::vec& X_offset,
                             const bool standardize) = 0;

  virtual double hessianTerm(const arma::sp_mat& X,
                             const arma::uword j,
                             const arma::vec& X_offset,
                             const bool standardize) = 0;

  virtual void updateGradientOfCorrelation(arma::vec& c_grad,
                                           const arma::mat& X,
                                           const arma::vec& Hinv_s,
                                           const arma::vec& s,
                                           const arma::uvec& active_set,
                                           const arma::uvec& restricted_set,
                                           const arma::vec& X_offset,
                                           const bool standardize) = 0;

  virtual void updateGradientOfCorrelation(arma::vec& c_grad,
                                           const arma::sp_mat& X,
                                           const arma::vec& Hinv_s,
                                           const arma::vec& s,
                                           const arma::uvec& active_set,
                                           const arma::uvec& restricted_set,
                                           const arma::vec& X_offset,
                                           const bool standardize) = 0;

  virtual void standardizeY(arma::vec& y) = 0;

  virtual double safeScreeningRadius(const double duality_gap,
                                     const double lambda) = 0;

  virtual double toleranceModifier(const arma::vec& y) = 0;
};

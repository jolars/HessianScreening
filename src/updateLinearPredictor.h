#pragma once

#include <RcppArmadillo.h>

inline void
updateLinearPredictor(arma::vec& Xbeta,
                      const arma::mat& X,
                      const arma::vec& beta,
                      const arma::vec& X_offset,
                      const bool standardize)
{
  Xbeta = X * beta;
}

inline void
updateLinearPredictor(arma::vec& Xbeta,
                      const arma::sp_mat& X,
                      const arma::vec& beta,
                      const arma::vec& X_offset,
                      const bool standardize)
{
  Xbeta = X * beta;

  if (standardize)
    Xbeta -= arma::dot(beta, X_offset);
}

inline void
updateLinearPredictor(arma::vec& Xbeta,
                      const arma::mat& X,
                      const arma::vec& beta,
                      const arma::vec& X_offset,
                      const bool standardize,
                      const arma::uvec& ind)
{
  Xbeta = X.cols(ind) * beta(ind);
}

inline void
updateLinearPredictor(arma::vec& Xbeta,
                      const arma::sp_mat& X,
                      const arma::vec& beta,
                      const arma::vec& X_offset,
                      const bool standardize,
                      const arma::uvec& ind)
{
  Xbeta = X.cols(ind) * beta(ind);

  if (standardize)
    Xbeta -= arma::dot(beta(ind), X_offset(ind));
}

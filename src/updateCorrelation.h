#pragma once

#include <RcppArmadillo.h>

inline void
updateCorrelation(arma::vec& c,
                  const arma::vec& residual,
                  const arma::mat& X,
                  const arma::vec& X_offset,
                  const bool standardize)
{
  for (arma::uword j = 0; j < c.n_elem; ++j) {
    c(j) = arma::dot(X.unsafe_col(j), residual);
  }
}

inline void
updateCorrelation(arma::vec& c,
                  const arma::vec& residual,
                  const arma::sp_mat& X,
                  const arma::vec& X_offset,
                  const bool standardize)
{
  for (arma::uword j = 0; j < c.n_elem; ++j) {
    c(j) = arma::dot(X.col(j), residual);
  }

  if (standardize) {
    c -= X_offset * arma::accu(residual);
  }
}

inline void
updateCorrelation(arma::vec& c,
                  const arma::vec& residual,
                  const arma::mat& X,
                  const arma::uvec& ind,
                  const arma::vec& X_offset,
                  const bool standardize)
{
  for (auto&& j : ind) {
    c(j) = arma::dot(X.unsafe_col(j), residual);
  }
}

inline void
updateCorrelation(arma::vec& c,
                  const arma::vec& residual,
                  const arma::sp_mat& X,
                  const arma::uvec& ind,
                  const arma::vec& X_offset,
                  const bool standardize)
{
  for (auto&& j : ind) {
    c(j) = arma::dot(X.col(j), residual);
  }

  if (standardize) {
    c(ind) -= X_offset(ind) * arma::accu(residual);
  }
}

inline void
updateCorrelation(arma::vec& c,
                  const arma::vec& residual,
                  const arma::mat& X,
                  const arma::uword j,
                  const arma::vec& X_offset,
                  const bool standardize)
{
  c(j) = arma::dot(X.unsafe_col(j), residual);
}

inline void
updateCorrelation(arma::vec& c,
                  const arma::vec& residual,
                  const arma::sp_mat& X,
                  const arma::uword j,
                  const arma::vec& X_offset,
                  const bool standardize)
{
  c(j) = arma::dot(X.col(j), residual);

  if (standardize) {
    c(j) -= X_offset(j) * arma::accu(residual);
  }
}

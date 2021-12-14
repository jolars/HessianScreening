#pragma once

#include <RcppArmadillo.h>

template<typename T>
arma::vec
colNormsSquared(const T& X)
{
  arma::vec out(X.n_cols);

  for (arma::uword j = 0; j < X.n_cols; ++j) {
    out(j) = std::pow(norm(X.col(j)), 2);
  }

  return out;
}

template<typename T>
arma::vec
colNorms(const T& X)
{
  arma::vec out(X.n_cols);

  for (arma::uword j = 0; j < X.n_cols; ++j) {
    out(j) = norm(X.col(j));
  }

  return out;
}

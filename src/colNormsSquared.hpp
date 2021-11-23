#pragma once

#include <RcppArmadillo.h>

template<typename T>
arma::vec
colNormsSquared(const T& X)
{
  using namespace arma;

  vec out(X.n_cols);

  for (uword j = 0; j < X.n_cols; ++j) {
    out(j) = std::pow(norm(X.col(j)), 2);
  }

  return out;
}

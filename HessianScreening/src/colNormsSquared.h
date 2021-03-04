#pragma once

#include <RcppArmadillo.h>

using namespace arma;

template<typename T>
vec
colNormsSquared(const T& X)
{
  vec out(X.n_cols);

#pragma omp parallel for
  for (uword j = 0; j < X.n_cols; ++j) {
    out(j) = std::pow(norm(X.col(j)), 2);
  }

  return out;
}

#include "squaredColNorms.h"

arma::vec
squaredColNorms(const arma::mat& X,
                const arma::vec& X_offset,
                const bool standardize)
{
  arma::vec out(X.n_cols);

  for (arma::uword j = 0; j < X.n_cols; ++j) {
    out(j) = accu(square(X.col(j)));
  }

  return out;
}

arma::vec
squaredColNorms(const arma::sp_mat& X,
                const arma::vec& X_offset,
                const bool standardize)
{
  using namespace arma;

  uword n = X.n_rows;
  uword p = X.n_cols;

  vec out(p);

  for (uword j = 0; j < X.n_cols; ++j) {
    if (standardize) {
      out(j) = accu(square(X.col(j))) - 2 * X_offset(j) * accu(X.col(j)) +
               n * std::pow(X_offset(j), 2);
    } else {
      out(j) = accu(square(X.col(j)));
    }
  }

  return out;
}

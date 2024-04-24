#include "standardize.h"

void
standardizeX(arma::vec& X_mean, arma::vec& X_sd, arma::mat& X)
{
  using namespace arma;

  const uword p = X.n_cols;

  for (uword j = 0; j < p; j++) {
    X_mean(j) = mean(X.col(j));
    X.col(j) -= X_mean(j);

    double X_sd_j = stddev(X.col(j), 1);

    if (X_sd_j != 0) {
      X_sd(j) = X_sd_j;
      X.col(j) /= X_sd_j;
    }
  }
}

void
standardizeX(arma::vec& X_mean, arma::vec& X_sd, arma::sp_mat& X)
{
  using namespace arma;

  const uword n = X.n_rows;
  const uword p = X.n_cols;

  for (uword j = 0; j < p; j++) {
    X_mean(j) = mean(X.col(j));

    double X_sd_j = norm(X.col(j) - accu(X.col(j)) / n) / std::sqrt(n);

    if (X_sd_j != 0) {
      X_sd(j) = X_sd_j;
      X.col(j) /= X_sd_j;
    }
  }
}

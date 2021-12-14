#include "rescaleCoefficients.h"

void
rescaleCoefficients(arma::mat& betas,
                    const arma::vec& X_mean,
                    const arma::vec& X_sd,
                    const double y_mean)
{
  const arma::uword p = betas.n_rows;

  for (arma::uword j = 0; j < p; ++j) {
    betas.row(j) /= X_sd(j);
  }
}

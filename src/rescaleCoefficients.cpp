#include "rescaleCoefficients.h"
#include <RcppArmadillo.h>

using namespace arma;

void
rescaleCoefficients(arma::mat& betas,
                    const arma::vec& X_mean,
                    const arma::vec& X_sd,
                    const double y_mean)
{
  for (arma::uword j = 0; j < betas.n_rows; ++j) {
    betas.row(j) /= X_sd(j);
  }
}

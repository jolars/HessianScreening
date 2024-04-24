#pragma once

#include <RcppArmadillo.h>

void
rescaleCoefficients(arma::mat& betas,
                    const arma::vec& X_mean,
                    const arma::vec& X_sd,
                    const double y_mean);

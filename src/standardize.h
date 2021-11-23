#pragma once

#include <RcppArmadillo.h>

void
standardizeX(arma::vec& X_mean, arma::vec& X_sd, arma::mat& X);

void
standardizeX(arma::vec& X_mean, arma::vec& X_sd, arma::sp_mat& X);

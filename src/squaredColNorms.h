#pragma once

#include <RcppArmadillo.h>

arma::vec
squaredColNorms(const arma::mat& X,
                const arma::vec& X_offset,
                const bool standardize);

arma::vec
squaredColNorms(const arma::sp_mat& X,
                const arma::vec& X_offset,
                const bool standardize);

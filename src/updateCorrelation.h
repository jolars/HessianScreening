#pragma once

#include <RcppArmadillo.h>

void
updateCorrelation(arma::vec& c,
                  const arma::vec& residual,
                  const arma::mat& X,
                  const arma::vec& offset,
                  const bool standardize);

void
updateCorrelation(arma::vec& c,
                  const arma::vec& residual,
                  const arma::sp_mat& X,
                  const arma::vec& offset,
                  const bool standardize);

void
updateCorrelation(arma::vec& c,
                  const arma::vec& residual,
                  const arma::mat& X,
                  const arma::uvec& ind,
                  const arma::vec& offset,
                  const bool standardize);

void
updateCorrelation(arma::vec& c,
                  const arma::vec& residual,
                  const arma::sp_mat& X,
                  const arma::uvec& ind,
                  const arma::vec& offset,
                  const bool standardize);

void
updateCorrelation(arma::vec& c,
                  const arma::vec& residual,
                  const arma::mat& X,
                  const arma::uword j,
                  const arma::vec& offset,
                  const bool standardize);

void
updateCorrelation(arma::vec& c,
                  const arma::vec& residual,
                  const arma::sp_mat& X,
                  const arma::uword j,
                  const arma::vec& offset,
                  const bool standardize);

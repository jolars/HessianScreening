#pragma once

#include <RcppArmadillo.h>

double
innerProduct(const arma::mat& X,
             const arma::uword j,
             const arma::vec& v,
             const arma::vec& X_offset,
             const bool standardize);

double
innerProduct(const arma::sp_mat& X,
             const arma::uword j,
             const arma::vec& v,
             const arma::vec& X_offset,
             const bool standardize);

double
weightedInnerProduct(const arma::mat& X,
                     const arma::uword j,
                     const arma::vec& v,
                     const arma::vec& weights,
                     const arma::vec& X_offset,
                     const bool standardize);

double
weightedInnerProduct(const arma::sp_mat& X,
                     const arma::uword j,
                     const arma::vec& v,
                     const arma::vec& weights,
                     const arma::vec& X_offset,
                     const bool standardize);

void
addScaledColumn(arma::vec& out,
                const arma::mat& X,
                const arma::uword j,
                const double a,
                const arma::vec& X_offset,
                const bool standardize);
void
addScaledColumn(arma::vec& out,
                const arma::sp_mat& X,
                const arma::uword j,
                const double a,
                const arma::vec& X_offset,
                const bool standardize);

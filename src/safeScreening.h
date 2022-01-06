#pragma once

#include "model.h"
#include "updateCorrelation.h"
#include <RcppArmadillo.h>

template<typename T>
void
safeScreening(arma::uvec& screened,
              arma::uvec& screened_set,
              arma::vec& c,
              arma::vec& residual,
              arma::vec& Xbeta,
              arma::vec& beta,
              const arma::vec& XTcenter,
              const double r_screen,
              const std::unique_ptr<Model>& model,
              const T& X,
              const arma::vec& y,
              const arma::vec& X_offset,
              const bool standardize,
              const arma::vec& X_norms_squared)
{
  using namespace arma;

  for (auto&& j : screened_set) {
    double r_normX_j = r_screen * std::sqrt(X_norms_squared(j));

    if (r_normX_j >= 1) {
      continue;
    }

    if (std::abs(XTcenter(j)) + r_normX_j + datum::eps < 1) {
      // predictor must be zero; update residual and remove from screened set
      if (beta(j) != 0) {
        model->adjustResidual(
          residual, Xbeta, X, y, j, -beta(j), X_offset, standardize);

        beta(j) = 0;
      }

      screened(j) = false;
    }
  }

  screened_set = find(screened);
}

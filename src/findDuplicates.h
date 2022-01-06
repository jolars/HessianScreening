#pragma once

#include "model.h"
#include "utils.h"
#include <RcppArmadillo.h>

template<typename T>
std::tuple<arma::uvec, arma::uvec>
findDuplicates(arma::uvec& active_set,
               arma::uvec& active_set_prev,
               const T& X,
               const std::unique_ptr<Model>& model,
               const arma::vec& X_offset,
               const bool standardize)
{
  using namespace arma;

  const uvec activate = setDiff(active_set, active_set_prev);

  std::vector<uword> originals, duplicates;

  if (!activate.is_empty()) {
    const mat D = model->hessian(X, activate, X_offset, standardize);

    for (uword i = 0; i < D.n_rows - 1; ++i) {
      if (!contains(duplicates, activate(i))) {
        for (uword j = (i + 1); j < D.n_rows; ++j) {
          if (std::sqrt(D(j, j) * D(i, i)) == std::abs(D(i, j))) {
            originals.emplace_back(activate(i));
            duplicates.emplace_back(activate(j));
          }
        }
      }
    }
  }

  return { conv_to<uvec>::from(originals), conv_to<uvec>::from(duplicates) };
}

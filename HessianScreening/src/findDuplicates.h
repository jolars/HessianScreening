#pragma once

#include <RcppArmadillo.h>

#include "model.h"
#include "utils.h"

using namespace arma;

template<typename T>
std::tuple<uvec, uvec>
findDuplicates(uvec& active_set,
               uvec& active_prev_set,
               const T& X,
               const std::unique_ptr<Model>& model)
{
  uvec activate = safeSetDiff(active_set, active_prev_set);

  std::vector<uword> originals, duplicates;

  if (!activate.is_empty()) {
    mat D = model->hessian(X, activate);

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
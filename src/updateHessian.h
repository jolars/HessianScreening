#pragma once

#include <RcppArmadillo.h>

#include "model.h"

using namespace arma;

template<typename T>
void
updateHessian(mat& H,
              mat& Hinv,
              uvec& active_set,
              uvec& active_set_prev,
              uvec& active_perm,
              uvec& active_perm_prev,
              const std::unique_ptr<Model>& model,
              const T& X,
              const bool verify_hessian,
              const uword verbosity);

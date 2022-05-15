#pragma once

#include "model.h"
#include <RcppArmadillo.h>
#include <memory>

std::unique_ptr<Model>
setupModel(const std::string family,
           const arma::vec& X_norms_squared,
           const arma::uword n,
           const std::string log_hessian_update_type);

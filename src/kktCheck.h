#pragma once

#include <RcppArmadillo.h>

void
kktCheck(arma::uvec& violations,
         arma::uvec& screened,
         const arma::vec& c,
         const arma::uvec& check_set,
         const double lambda);

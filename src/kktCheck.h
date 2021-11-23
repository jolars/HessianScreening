#pragma once

#include <RcppArmadillo.h>

double
kktCheck(arma::uvec& violations,
         arma::uvec& screened,
         const arma::vec& c,
         const arma::uvec& check_set,
         const double lambda);

double
kktCheck(arma::uvec& violations,
         const arma::vec& c,
         const arma::uvec& check_set,
         const double lambda);

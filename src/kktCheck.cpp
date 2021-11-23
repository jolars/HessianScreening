#include "kktCheck.h"
#include <RcppArmadillo.h>

double
kktCheck(arma::uvec& violations,
         arma::uvec& screened,
         const arma::vec& c,
         const arma::uvec& check_set,
         const double lambda)
{
  double scale = lambda;

  for (auto&& j : check_set) {
    double c_j = std::abs(c(j));
    if (c_j >= lambda) {
      if (c_j > scale)
        scale = c_j;
      violations[j] = true;
      screened[j] = true;
    }
  }

  return scale;
}

double
kktCheck(arma::uvec& violations,
         const arma::vec& c,
         const arma::uvec& check_set,
         const double lambda)
{
  double scale = lambda;

  for (auto&& j : check_set) {
    double c_j = std::abs(c(j));
    if (c_j >= lambda) {
      if (c_j > scale)
        scale = c_j;
      violations[j] = true;
    }
  }

  return scale;
}

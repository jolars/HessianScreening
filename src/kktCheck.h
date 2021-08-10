#pragma once

#include <RcppArmadillo.h>

using namespace arma;

double
kktCheck(uvec& violations,
         uvec& screened,
         const vec& c,
         const uvec& check_set,
         const double lambda)
{
  double scale = lambda;
  for (auto&& j : check_set) {
    double c_j = std::abs(c(j)) ;
    if (c_j >= lambda) {
      if(c_j > scale)
        scale = c_j;
      violations[j] = true;
      screened[j] = true;
    }
  }
  return(scale);
}

double
  kktCheck(uvec& violations,
           const vec& c,
           const uvec& check_set,
           const double lambda)
{
    double scale = lambda;
    for (auto&& j : check_set) {
      double c_j = std::abs(c(j)) ;
      if (c_j >= lambda) {
        if(c_j > scale)
          scale = c_j;
        violations[j] = true;
      }
    }
    return(scale);
}


#include "kktCheck.h"

void
kktCheck(arma::uvec& violations,
         arma::uvec& screened,
         const arma::vec& c,
         const arma::uvec& check_set,
         const double lambda)
{
  for (auto&& j : check_set) {
    if (std::abs(c(j)) >= lambda) {
      violations[j] = true;
      screened[j] = true;
    }
  }
}

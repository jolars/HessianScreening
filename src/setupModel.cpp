#include "setupModel.h"
#include "binomial.h"
#include "gaussian.h"
#include "poisson.h"

std::unique_ptr<Model>
setupModel(const std::string family,
           const arma::vec& X_norms_squared,
           const arma::uword n,
           const std::string log_hessian_update_type)
{
  if (family == "binomial")
    return std::make_unique<Binomial>(family, n, log_hessian_update_type);
  else if (family == "poisson")
    return std::make_unique<Poisson>(family, n);

  return std::make_unique<Gaussian>(family, X_norms_squared);
}

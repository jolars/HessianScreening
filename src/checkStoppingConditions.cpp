#include "checkStoppingConditions.h"

bool
checkStoppingConditions(const arma::uword step,
                        const arma::uword n,
                        const arma::uword p,
                        const arma::uword n_lambda,
                        const arma::uword n_active,
                        const double lambda,
                        const double lambda_min,
                        const double dev,
                        const double dev_prev,
                        const double null_dev,
                        const std::string screening_type,
                        const arma::uword verbosity)
{
  if (step == 1) {
    // first step; always continue
    return false;
  }

  if (verbosity >= 1) {
    Rprintf("  checking stopping conditions\n");
  }

  double dev_ratio = 1.0 - dev / null_dev;
  double dev_change = 1.0 - dev / dev_prev;

  if (verbosity >= 1) {
    Rprintf(
      "    dev ratio:  %.3f\n    dev change: %.6f\n", dev_ratio, dev_change);
  }

  if (dev_change <= 1e-5) {
    return true;
  }

  if (dev_ratio >= 0.999) {
    return true;
  }

  if (lambda <= lambda_min) {
    return true;
  }

  if (n <= p && n_active >= n) {
    return true;
  }

  if (step >= n_lambda) {
    return true;
  }

  return false;
}

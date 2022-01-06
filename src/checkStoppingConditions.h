#pragma once

#include <RcppArmadillo.h>

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
                        const arma::uword verbosity);

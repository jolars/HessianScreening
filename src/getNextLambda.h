#pragma once

#include <RcppArmadillo.h>

struct GetNextLambda
{
  const arma::vec& beta;
  const arma::vec& c;
  const arma::vec& c_grad;
  const arma::vec& lambda_grid;
  const double lambda_min;
  const double lambda_min_step;
  const std::string screening_type;
  arma::uword n_target_nonzero;
  double n_target_nonzero_mod{ 1 };
  const arma::uword verbosity;

  double t{ 1 };
  double eta{ 0.01 };

  std::vector<double> ts = { 1.0 };
  std::vector<double> n_target_nonzero_mods = { 1.0 };

  GetNextLambda(const arma::vec& beta,
                const arma::vec& c,
                const arma::vec& c_grad,
                const arma::vec& lambda_grid,
                const double lambda_min,
                const double lambda_min_step,
                const std::string screening_type,
                const arma::uword n_target_nonzero,
                const arma::uword verbosity);

  double operator()(const arma::vec& Hinv_s,
                    const double lambda,
                    const arma::uvec& active,
                    const arma::uword n_new_active,
                    const arma::uword step_number);
};

#pragma once

#include "prox.h"
#include "updateCorrelation.h"
#include <RcppArmadillo.h>

class Model
{
public:
  const std::string family;

  arma::vec& y;
  arma::vec& beta;
  arma::vec& residual;
  arma::vec& Xbeta;

  const arma::vec& X_mean_scaled;
  const arma::vec& X_norms_squared;

  const arma::uword n;
  const arma::uword p;
  const bool standardize;

  double dual_scale{ 0 };

  Model(const std::string family,
        arma::vec& y,
        arma::vec& beta,
        arma::vec& residual,
        arma::vec& Xbeta,
        const arma::vec& X_mean_scaled,
        const arma::vec& X_norms_squared,
        const arma::uword n,
        const arma::uword p,
        const bool standardize);

  virtual ~Model();

  void setLogHessianUpdateType(const std::string new_log_hessian_update_type);

  virtual double primal(const double lambda) = 0;

  virtual double primal(const double lambda,
                        const arma::uvec& screened_set) = 0;

  virtual double dual() = 0;

  virtual double scaledDual(const double lambda) = 0;

  virtual double deviance() = 0;

  virtual arma::mat hessian(const arma::mat& X, const arma::uvec& ind) = 0;

  virtual arma::mat hessian(const arma::sp_mat& X, const arma::uvec& ind) = 0;

  virtual arma::mat hessianUpperRight(const arma::mat& X,
                                      const arma::uvec& ind_a,
                                      const arma::uvec& ind_b) = 0;

  virtual arma::mat hessianUpperRight(const arma::sp_mat& X,
                                      const arma::uvec& ind_a,
                                      const arma::uvec& ind_b) = 0;

  virtual double hessianTerm(const arma::mat& X, const arma::uword j) = 0;

  virtual double hessianTerm(const arma::sp_mat& X, const arma::uword j) = 0;

  virtual void updateResidual() = 0;

  virtual void adjustResidual(const arma::mat& X,
                              const arma::uword j,
                              const double beta_diff) = 0;

  virtual void adjustResidual(const arma::sp_mat& X,
                              const arma::uword j,
                              const double beta_diff) = 0;

  virtual void updateGradientOfCorrelation(
    arma::vec& c_grad,
    const arma::mat& X,
    const arma::vec& Hinv_s,
    const arma::vec& s,
    const arma::uvec& active_set,
    const arma::uvec& restricted_set) = 0;

  virtual void updateGradientOfCorrelation(
    arma::vec& c_grad,
    const arma::sp_mat& X,
    const arma::vec& Hinv_s,
    const arma::vec& s,
    const arma::uvec& active_set,
    const arma::uvec& restricted_set) = 0;

  virtual void standardizeY() = 0;

  virtual double safeScreeningRadius(const double duality_gap,
                                     const double lambda) = 0;

  void updateLinearPredictor(const arma::mat& X);

  void updateLinearPredictor(const arma::sp_mat& X);

  void updateLinearPredictor(const arma::mat& X, const arma::uvec& ind);

  void updateLinearPredictor(const arma::sp_mat& X, const arma::uvec& ind);

  template<typename T>
  void safeScreening(arma::uvec& screened,
                     arma::uvec& screened_set,
                     const T& X,
                     const arma::vec& XTcenter,
                     const double r_screen)
  {
    using namespace arma;

    for (auto&& j : screened_set) {
      double r_normX_j = r_screen * std::sqrt(X_norms_squared(j));

      if (r_normX_j >= 1) {
        continue;
      }

      if (std::abs(XTcenter(j)) + r_normX_j + std::sqrt(datum::eps) < 1) {
        // predictor must be zero; update residual and remove from screened set
        if (beta(j) != 0) {
          adjustResidual(X, j, -beta(j));
          beta(j) = 0;
        }

        screened(j) = false;
      }
    }

    screened_set = find(screened);
  }

  template<typename T>
  std::tuple<double, double, double, arma::uword, double> fit(
    arma::uvec& screened,
    arma::vec& c,
    const arma::uvec& active_set,
    const T& X,
    const arma::vec& X_norms_squared,
    const double lambda,
    const double lambda_prev,
    const double lambda_max,
    const double null_primal,
    const arma::uword n_active_prev,
    const std::string screening_type,
    const bool gap_safe_active_start,
    const bool first_run,
    const arma::uword step,
    const arma::uword maxit,
    const double tol_gap_rel,
    const int line_search,
    const arma::uword ws_size_init,
    const arma::uword verbosity)
  {
    using namespace arma;

    const uword p = X.n_cols;

    if (screening_type == "celer" || screening_type == "gap_safe")
      screened.fill(true);

    uvec screened_set = find(screened);
    uvec working_set = screened_set;

    double primal_value = primal(lambda, screened_set);
    double dual_value = dual();
    double duality_gap = primal_value - dual_value;
    double duality_gap_rel = duality_gap / std::max(1.0, primal_value);

    dual_scale = std::max(lambda, max(abs(c)));

    // celer parameters
    double duality_gap_inner = duality_gap;
    double dual_value_inner = duality_gap;
    double dual_value_old = duality_gap;
    double dual_value_res = duality_gap;
    uword ws_size = n_active_prev == 0 ? ws_size_init : n_active_prev;
    bool inner_solver_converged = false;
    vec c_inner(p);
    vec c_old(p);

    // dual variables
    vec theta(p);
    vec theta_inner(p);
    vec theta_old(p);

    // line search parameters
    const double a = 0.1;
    const double b = 0.5;

    vec XTcenter(p);

    vec t(p, fill::ones); // learning rates

    const uword screen_interval = 10;

    double n_screened = 0;
    uword it = 0;

    if (!screened_set.is_empty()) {
      updateLinearPredictor(X, screened_set);
      updateResidual();

      while (it < maxit) {
        if (verbosity >= 2) {
          Rprintf("    iter: %i\n", it + 1);
        }

        if (screening_type == "gap_safe" && it % screen_interval == 0) {
          if (it > 0) {
            updateLinearPredictor(X, screened_set);
            updateResidual();
          }
          updateCorrelation(c, residual, X, screened_set, X_mean_scaled, standardize);

          dual_scale = std::max(lambda, max(abs(c)));

          primal_value = primal(lambda, screened_set);
          dual_value = scaledDual(lambda);
          duality_gap = primal_value - dual_value;
          duality_gap_rel = duality_gap / std::max(1.0, primal_value);

          if (verbosity >= 2)
            Rprintf("      duality gap: %f\n", duality_gap_rel);

          if (duality_gap_rel <= tol_gap_rel)
            break;

          // screening
          XTcenter = c / dual_scale;
          double r_screen =
            safeScreeningRadius(std::max(duality_gap, 0.0), lambda);

          safeScreening(screened, screened_set, X, XTcenter, r_screen);
        }

        if (screening_type == "celer") {
          if (inner_solver_converged || it == 0) {
            vec residual_inner = residual / dual_scale;
            c_inner = c;
            dual_value_inner = dual_value;

            if (it > 0) {
              updateLinearPredictor(X, screened_set);
              updateResidual();
            }
            updateCorrelation(c, residual, X, screened_set, X_mean_scaled, standardize);

            primal_value = primal(lambda);

            dual_scale = std::max(lambda, max(abs(c(screened_set))));
            theta = residual / dual_scale;
            dual_value = scaledDual(lambda);

            // if (it > 0) {
            //   if (dual_value_inner > dual_value &&
            //       dual_value_inner > dual_value_old) {
            //     dual_value = dual_value_inner;
            //     theta = theta_inner;
            //     c = c_inner;
            //   } else if (dual_value_old > dual_value &&
            //              dual_value_old > dual_value_inner) {
            //     dual_value = dual_value_old;
            //     theta = theta_old;
            //     c = c_old;
            //   }
            // }

            theta_old = theta;
            c_old = c;
            dual_value_old = dual_value;

            duality_gap = primal_value - dual_value;
            duality_gap_rel = duality_gap / std::max(1.0, primal_value);

            if (verbosity >= 2)
              Rprintf("      duality gap: %f\n", duality_gap_rel);

            if (duality_gap_rel <= tol_gap_rel)
              break;

            vec d = (1.0 - c(screened_set) / dual_scale) /
                    sqrt(X_norms_squared(screened_set));

            if (it == 0) {
              working_set = active_set;
            }

            d(working_set).fill(-1);

            d.print();

            if (it > 0) {
              ws_size *= 2;
            }

            Rcpp::Rcout << "ws_size: " << ws_size << std::endl;

            ws_size = std::min(ws_size, screened_set.n_elem);

            Rcpp::Rcout << "ws_size: " << ws_size << std::endl;

            //             XTcenter = c / dual_scale;
            //             double r_screen =
            //               safeScreeningRadius(std::max(duality_gap, 0.0),
            //               lambda);

            //             safeScreening(screened, screened_set, X, XTcenter,
            //             r_screen);

            uvec ind = sort_index(d, "ascend");
            working_set = screened_set(ind.head(ws_size));
            // screened.zeros();
            // screened(screened_set).ones();

            working_set.print();

            inner_solver_converged = false;
          }
        }

        n_screened += screened_set.n_elem;

        if (line_search == 3) {
          for (auto&& j : working_set) {
            updateCorrelation(c, residual, X, j, X_mean_scaled, standardize);
            double hess_j = hessianTerm(X, j);

            double beta_j_old = beta(j);
            double v =
              prox(beta_j_old + c(j) / hess_j, lambda / hess_j) - beta(j);

            if (v != 0) {
              if (family == "binomial" && line_search > 0) {
                // line search (see J. D. Lee, Y. Sun, and M. A. Saunders,
                // “Proximal Newton-type methods for minimizing composite
                // functions,” arXiv:1206.1623 [cs, math, stat], Mar. 2014,
                // Accessed: Jan. 12, 2020. [Online]. Available:
                // http://arxiv.org/abs/1206.1623)

                double primal_value_old = primal(lambda, working_set);

                while (true) {
                  double beta_j_prev = beta(j);

                  beta(j) = beta_j_old + t(j) * v;
                  adjustResidual(X, j, beta(j) - beta_j_prev);

                  primal_value = primal(lambda, working_set);

                  double eta = -c(j) * v + lambda * (std::abs(beta_j_old + v) -
                                                     std::abs(beta_j_old));

                  if (primal_value * (1 - std::sqrt(datum::eps)) <=
                      primal_value_old + a * t(j) * eta) {
                    break;
                  } else {
                    t(j) *= b;
                  }

                  Rcpp::checkUserInterrupt();
                }
              } else {
                beta(j) = beta_j_old + v;
                adjustResidual(X, j, beta(j) - beta_j_old);
              }
            }
          }

        } else {
          for (auto&& j : working_set) {

            updateCorrelation(c, residual, X, j, X_mean_scaled, standardize);
            double hess_j = hessianTerm(X, j);

            double beta_j_old = beta(j);
            double v =
              prox(beta_j_old + c(j) / hess_j, lambda / hess_j) - beta(j);

            if (v != 0) {
              if (family == "binomial" && line_search > 0) {
                // line search
                bool do_line_search = false;
                double primal_value_old;

                if (line_search == 1) {
                  primal_value_old = primal(lambda, working_set);
                  do_line_search = true;
                }

                // line search (see J. D. Lee, Y. Sun, and M. A. Saunders,
                // “Proximal Newton-type methods for minimizing composite
                // functions,” arXiv:1206.1623 [cs, math, stat], Mar. 2014,
                // Accessed: Jan. 12, 2020. [Online]. Available:
                // http://arxiv.org/abs/1206.1623)
                beta(j) = beta_j_old + t(j) * v;
                double c_j_old = c(j);
                double eta = -c(j) * v + lambda * (std::abs(beta_j_old + v) -
                                                   std::abs(beta_j_old));
                adjustResidual(X, j, beta(j) - beta_j_old);
                double beta_j_prev = beta(j);

                if (line_search == 2) {
                  updateCorrelation(c, residual, X, j, X_mean_scaled, standardize);
                  if (std::max(c_j_old, c(j)) - std::min(c_j_old, c(j)) >
                      lambda_prev - lambda) {
                    if (verbosity >= 2) {
                      Rprintf(
                        "    linesearch type 2 at iter: %i, index: %i t: %e\n",
                        it + 1,
                        j,
                        t(j));
                    }
                    do_line_search = true;
                    adjustResidual(X, j, beta_j_old - beta(j));
                    beta(j) = beta_j_old;
                    primal_value_old = primal(lambda, working_set);
                    beta(j) = beta_j_old + t(j) * v;
                    adjustResidual(X, j, beta(j) - beta_j_old);
                  }
                }

                while (do_line_search) {
                  primal_value = primal(lambda, working_set);

                  if (primal_value * (1 - std::sqrt(datum::eps)) <=
                      primal_value_old + a * t(j) * eta) {
                    break;
                  } else {
                    t(j) *= b;
                  }
                  beta(j) = beta_j_old + t(j) * v;
                  adjustResidual(X, j, beta(j) - beta_j_prev);
                  beta_j_prev = beta(j);
                  Rcpp::checkUserInterrupt();
                }
                if (t(j) < 1) {
                  if (line_search < 3)
                    t(j) /= b;
                }

              } else {
                beta(j) = beta_j_old + v;
                adjustResidual(X, j, beta(j) - beta_j_old);
              }
            }
          }
        }

        it++;

        if (screening_type != "gap_safe") {
          primal_value = primal(lambda, working_set);
          updateCorrelation(c, residual, X, working_set, X_mean_scaled, standardize);
          dual_scale = std::max(lambda, max(abs(c(working_set))));
          dual_value = scaledDual(lambda);
          duality_gap = primal_value - dual_value;
          duality_gap_rel = duality_gap / std::max(1.0, primal_value);

          if (verbosity >= 2)
            Rprintf("      duality gap: %f\n", duality_gap_rel);

          if (duality_gap_rel <= tol_gap_rel) {
            inner_solver_converged = true;
          }

          if (inner_solver_converged && screening_type != "celer") {
            break;
          }
        }

        Rcpp::checkUserInterrupt();
      }
    } else {
      beta.zeros();
    }

    double avg_screened = n_screened / (it + 1);

    return { primal_value, dual_value, duality_gap, it + 1, avg_screened };
  }
};

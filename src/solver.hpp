#pragma once

#include "model.h"
#include "safeScreening.h"
#include "updateCorrelation.h"
#include "updateLinearPredictor.h"
#include "utils.h"
#include <RcppArmadillo.h>

template<typename T>
std::tuple<double, double, double, arma::uword, double>
fit(arma::uvec& screened,
    arma::vec& c,
    arma::vec& residual,
    arma::vec& Xbeta,
    arma::vec& beta,
    const std::unique_ptr<Model>& model,
    const T& X,
    const arma::vec& y,
    const arma::vec& X_norms_squared,
    const arma::vec& X_offset,
    const bool standardize,
    const arma::uvec& active_set,
    const double lambda,
    const double lambda_prev,
    const double lambda_max,
    const arma::uword n_active_prev,
    const std::string screening_type,
    const bool celer_use_old_dual,
    const bool celer_use_accel,
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

  const uword n = X.n_rows;
  const uword p = X.n_cols;

  const uword check_frequency = 10;

  if (screening_type == "celer" || screening_type == "gap_safe")
    screened.fill(true);

  uvec screened_set = find(screened);
  uvec working_set = screened_set;

  double primal_value =
    model->primal(residual, Xbeta, beta, y, lambda, screened_set);

  double dual_scale = std::max(lambda, max(abs(c)));
  vec theta = residual / dual_scale;
  double dual_value = model->dual(theta, y, lambda);

  double duality_gap = primal_value - dual_value;
  double duality_gap_rel = duality_gap / std::max(1.0, primal_value);

  double duality_gap_rel_prev = duality_gap_rel;

  // celer parameters
  uword ws_size = ws_size_init;
  bool inner_solver_converged = true;
  vec d(p);
  vec residual_old;

  // celer acceleration
  const uword K = 5;
  vec residual_prev;
  vec residual_accel;
  vec z(K);
  vec celer_c(K);
  mat residual_storage(n, K);
  mat U(n, K);

  if (screening_type == "celer") {
    residual_old = residual;
    residual_prev = residual;
    residual_accel = residual;
  }

  // line search parameters
  const double a = 0.1;
  const double b = 0.5;

  vec XTcenter(p);

  vec t(p, fill::ones); // learning rates

  const uword screen_frequency = 10;

  double n_screened = 0;
  uword it = 0;

  if (!screened_set.is_empty()) {
    updateLinearPredictor(Xbeta, X, beta, X_offset, standardize, screened_set);
    model->updateResidual(residual, Xbeta, y);

    while (it < maxit) {
      if (verbosity >= 2) {
        Rprintf("    iter: %i\n", it + 1);
      }

      if (screening_type == "gap_safe" && it % screen_frequency == 0) {
        if (it > 0) {
          updateLinearPredictor(
            Xbeta, X, beta, X_offset, standardize, screened_set);
          model->updateResidual(residual, Xbeta, y);
        }
        updateCorrelation(c, residual, X, screened_set, X_offset, standardize);

        double primal_value =
          model->primal(residual, Xbeta, beta, y, lambda, screened_set);

        dual_scale = std::max(lambda, max(abs(c(screened_set))));
        theta = residual / dual_scale; // feasible dual point
        dual_value = model->dual(theta, y, lambda);

        duality_gap = primal_value - dual_value;
        duality_gap_rel = duality_gap / std::max(1.0, primal_value);

        if (verbosity >= 2)
          Rprintf("      global primal: %f, global dual: %f, global gap: %f\n",
                  primal_value,
                  dual_value,
                  duality_gap_rel);

        if (duality_gap_rel <= tol_gap_rel)
          break;

        // screening
        XTcenter = c / dual_scale;
        double r_screen =
          model->safeScreeningRadius(std::max(duality_gap, 0.0), lambda);

        safeScreening(screened,
                      screened_set,
                      c,
                      residual,
                      Xbeta,
                      beta,
                      XTcenter,
                      r_screen,
                      model,
                      X,
                      y,
                      X_offset,
                      standardize,
                      X_norms_squared);
      }

      if (screening_type == "celer") {
        if (inner_solver_converged) {
          primal_value =
            model->primal(residual, Xbeta, beta, y, lambda, screened_set);

          updateCorrelation(
            c, residual, X, screened_set, X_offset, standardize);

          dual_scale = std::max(lambda, max(abs(c(screened_set))));
          theta = residual / dual_scale;
          dual_value = model->dual(theta, y, lambda);

          if (it > 0 && celer_use_old_dual) {
            // check if dual point from previous check performs better
            vec c_old(p, fill::zeros);

            updateCorrelation(
              c_old, residual_old, X, screened_set, X_offset, standardize);

            double dual_scale_old =
              std::max(lambda, max(abs(c_old(screened_set))));
            vec theta_old = residual_old / dual_scale_old;
            double dual_value_old = model->dual(theta_old, y, lambda);

            if (dual_value_old > dual_value) {
              if (verbosity >= 2) {
                Rprintf("      old dual point performs better, use it\n");
              }

              dual_value = dual_value_old;
              c = c_old;
              theta = theta_old;
              dual_scale = dual_scale_old;
            }
          } 

          duality_gap = primal_value - dual_value;
          duality_gap_rel = duality_gap / std::max(1.0, primal_value);

          if (verbosity >= 2) {
            Rprintf(
              "      global primal: %f, global dual: %f, global rel_gap: %f\n",
              primal_value,
              dual_value,
              duality_gap_rel);
          }

          if (duality_gap_rel <= tol_gap_rel) {
            break;
          }

          vec d(p);
          d.fill(datum::inf);

          d(screened_set) = (1.0 - c(screened_set) / dual_scale) /
                            sqrt(X_norms_squared(screened_set));

          if (it == 0) {
            working_set = active_set;
          }

          d(working_set).fill(-1);

          // d.print();

          if (it > 0) {
            ws_size *= 2;
          }

          XTcenter = c / dual_scale;
          double r_screen =
            model->safeScreeningRadius(std::max(duality_gap, 0.0), lambda);

          safeScreening(screened,
                        screened_set,
                        c,
                        residual,
                        Xbeta,
                        beta,
                        XTcenter,
                        r_screen,
                        model,
                        X,
                        y,
                        X_offset,
                        standardize,
                        X_norms_squared);

          ws_size = std::min(ws_size, screened_set.n_elem);

          uvec ind = sort_index(d(screened_set), "ascend");

          working_set = screened_set(ind.head(ws_size));

          residual_old = residual;
        }
      }

      n_screened += screened_set.n_elem;

      working_set = shuffle(working_set);

      if (line_search == 3) {
        for (auto&& j : working_set) {
          updateCorrelation(c, residual, X, j, X_offset, standardize);
          double hess_j = model->hessianTerm(X, j, X_offset, standardize);

          double beta_j_old = beta(j);
          double v =
            prox(beta_j_old + c(j) / hess_j, lambda / hess_j) - beta(j);

          if (v != 0) {
            if (model->family == "binomial" && line_search > 0) {
              // line search (see J. D. Lee, Y. Sun, and M. A. Saunders,
              // “Proximal Newton-type methods for minimizing composite
              // functions,” arXiv:1206.1623 [cs, math, stat], Mar. 2014,
              // Accessed: Jan. 12, 2020. [Online]. Available:
              // http://arxiv.org/abs/1206.1623)

              double primal_value_old =
                model->primal(residual, Xbeta, beta, y, lambda, working_set);

              while (true) {
                double beta_j_prev = beta(j);

                beta(j) = beta_j_old + t(j) * v;
                model->adjustResidual(residual,
                                      Xbeta,
                                      X,
                                      y,
                                      j,
                                      beta(j) - beta_j_prev,
                                      X_offset,
                                      standardize);

                primal_value =
                  model->primal(residual, Xbeta, beta, y, lambda, working_set);

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
              model->adjustResidual(residual,
                                    Xbeta,
                                    X,
                                    y,
                                    j,
                                    beta(j) - beta_j_old,
                                    X_offset,
                                    standardize);
            }
          }
        }

      } else {
        for (auto&& j : working_set) {

          updateCorrelation(c, residual, X, j, X_offset, standardize);
          double hess_j = model->hessianTerm(X, j, X_offset, standardize);
          double beta_j_old = beta(j);
          double v =
            prox(beta_j_old + c(j) / hess_j, lambda / hess_j) - beta(j);

          if (v != 0) {
            if (model->family == "binomial" && line_search > 0) {
              // line search
              bool do_line_search = false;
              double primal_value_old;

              if (line_search == 1) {
                primal_value_old =
                  model->primal(residual, Xbeta, beta, y, lambda, working_set);
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
              model->adjustResidual(residual,
                                    Xbeta,
                                    X,
                                    y,
                                    j,
                                    beta(j) - beta_j_old,
                                    X_offset,
                                    standardize);

              double beta_j_prev = beta(j);

              if (line_search == 2) {
                updateCorrelation(c, residual, X, j, X_offset, standardize);
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
                  model->adjustResidual(residual,
                                        Xbeta,
                                        X,
                                        y,
                                        j,
                                        beta(j) - beta_j_old,
                                        X_offset,
                                        standardize);
                  beta(j) = beta_j_old;

                  primal_value_old = model->primal(
                    residual, Xbeta, beta, y, lambda, working_set);
                  beta(j) = beta_j_old + t(j) * v;
                  model->adjustResidual(residual,
                                        Xbeta,
                                        X,
                                        y,
                                        j,
                                        beta(j) - beta_j_old,
                                        X_offset,
                                        standardize);
                }
              }

              while (do_line_search) {
                primal_value =
                  model->primal(residual, Xbeta, beta, y, lambda, working_set);

                if (primal_value * (1 - std::sqrt(datum::eps)) <=
                    primal_value_old + a * t(j) * eta) {
                  break;
                } else {
                  t(j) *= b;
                }
                beta(j) = beta_j_old + t(j) * v;
                model->adjustResidual(residual,
                                      Xbeta,
                                      X,
                                      y,
                                      j,
                                      beta(j) - beta_j_prev,
                                      X_offset,
                                      standardize);
                beta_j_prev = beta(j);
                Rcpp::checkUserInterrupt();
              }
              if (t(j) < 1) {
                if (line_search < 3)
                  t(j) /= b;
              }

            } else {
              beta(j) = beta_j_old + v;
              model->adjustResidual(residual,
                                    Xbeta,
                                    X,
                                    y,
                                    j,
                                    beta(j) - beta_j_old,
                                    X_offset,
                                    standardize);
            }
          }
        }
      }

      if (screening_type == "celer" && celer_use_accel) {
        U = join_horiz(U.tail_cols(K - 1), residual - residual_prev);

        residual_storage =
          join_horiz(residual_storage.tail_cols(K - 1), residual);
        residual_prev = residual;
      }

      it++;

      if (screening_type != "gap_safe" && it % check_frequency == 0) {
        primal_value =
          model->primal(residual, Xbeta, beta, y, lambda, working_set);

        updateCorrelation(c, residual, X, working_set, X_offset, standardize);

        dual_scale = std::max(lambda, max(abs(c(working_set))));
        theta = residual / dual_scale;
        dual_value = model->dual(theta, y, lambda);

        duality_gap = primal_value - dual_value;
        duality_gap_rel = duality_gap / std::max(1.0, primal_value);

        if (screening_type == "celer" && celer_use_accel && it >= K) {
          // use dual extrapolation
          bool success =
            solve(z, symmatu(U.t() * U), ones<vec>(K), solve_opts::no_approx);

          if (success) {
            // if solver succeeds (well-conditioned problem), use acceleration
            celer_c = z / accu(z);

            residual_accel.zeros();
            for (uword i = 0; i < K; ++i) {
              residual_accel += celer_c(i) * residual_storage.col(K - i - 1);
            }

            vec c_accel(p, fill::zeros);

            updateCorrelation(
              c_accel, residual_accel, X, working_set, X_offset, standardize);

            double dual_scale_accel =
              std::max(1.0, std::max(lambda, max(abs(c_accel(working_set)))));
            vec theta_accel = residual_accel / dual_scale_accel;
            double dual_value_accel = model->dual(theta_accel, y, lambda);

            if (dual_value_accel > dual_value) {
              dual_value = dual_value_accel;
              c = c_accel;
              theta = theta_accel;
              residual = residual_accel;
            }
          }
        }

        if (verbosity >= 2)
          Rprintf("      primal: %f, dual: %f, rel_gap: %f\n",
                  primal_value,
                  dual_value,
                  duality_gap_rel);

        inner_solver_converged = duality_gap_rel <= tol_gap_rel;

        if (inner_solver_converged && screening_type != "celer") {
          break;
        }

        if (duality_gap_rel >= duality_gap_rel_prev) {
          if (verbosity >= 2) {
            Rprintf("      no progress; shuffling indices\n");
          }
          working_set = shuffle(working_set);
        }

        duality_gap_rel_prev = duality_gap_rel;
      }

      Rcpp::checkUserInterrupt();
    }
  } else {
    beta.zeros();
  }

  double avg_screened = n_screened / (it + 1);

  return { primal_value, dual_value, duality_gap, it + 1, avg_screened };
}

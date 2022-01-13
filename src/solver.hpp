#pragma once

#include "linearAlgebra.h"
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
    const bool shuffle,
    const bool celer_use_old_dual,
    const bool celer_use_accel,
    const bool celer_prune,
    const bool gap_safe_active_start,
    const bool first_run,
    const arma::uword step,
    const arma::uword maxit,
    const double tol_gap_rel,
    const bool line_search,
    const arma::uword ws_size_init,
    const arma::uword verbosity)
{
  using namespace arma;

  const uword n = X.n_rows;
  const uword p = X.n_cols;

  // const uword check_frequency = screening_type == "hessian" ? 1 : 10;
  const uword CHECK_FREQUENCY = n > p ? 2 : 10;
  const uword SCREEN_FREQUENCY = 10;

  if (screening_type == "celer" || screening_type == "gap_safe" ||
      screening_type == "blitz")
    screened.fill(true);

  uvec screened_set = find(screened);
  uvec working_set = screened_set;

  double primal_value =
    model->primal(residual, Xbeta, beta, y, lambda, screened_set);

  double dual_scale = std::max(lambda, max(abs(c)));
  vec theta = residual / dual_scale;
  double dual_value = model->dual(theta, y, lambda);

  vec w(n, fill::ones);

  const double GAP_EPS = 1.0;

  double duality_gap = primal_value - dual_value;
  double duality_gap_rel = duality_gap / std::max(GAP_EPS, primal_value);

  double duality_gap_rel_prev = duality_gap_rel;

  // blitz and celer parameters
  bool inner_solver_converged = true;
  bool progress = true;
  double dual_scale_old = dual_scale;
  double dual_value_old = dual_value;
  double primal_value_prev = primal_value;
  uword ws_size = ws_size_init;
  vec c_old(p);
  vec d(p);
  vec residual_old;

  // celer acceleration
  const uword K = 5;
  mat U(n, K);
  mat residual_storage(n, K);
  vec celer_c(K);
  vec residual_prev;
  vec z(K);

  // blitz parameters
  const uword MAX_BACKTRACK_ITR = 20;
  const uword MAX_PROX_NEWTON_CD_ITR = 20;
  const double PROX_NEWTON_EPSILON_RATIO = 10;
  const uword MIN_PROX_NEWTON_CD_ITR = 2;
  bool first_prox_newton_iteration = true;
  double prox_newton_grad_diff = 0;

  if (screening_type == "celer") {
    residual_old = residual;
    residual_prev = residual;
  }

  // test objects
  vec residual_test = residual;

  vec XTcenter(p);

  vec t(p, fill::ones); // learning rates

  double n_screened = 0;
  uword it = 0;
  double tol_gap_rel_inner = tol_gap_rel;

  if (!screened_set.is_empty()) {
    updateLinearPredictor(Xbeta, X, beta, X_offset, standardize, screened_set);
    model->updateResidual(residual, Xbeta, y);

    while (it < maxit) {
      if (verbosity >= 2) {
        Rprintf("    iter: %i\n", it + 2);
      }

      if (screening_type == "gap_safe" && it % SCREEN_FREQUENCY == 0) {
        updateCorrelation(c, residual, X, screened_set, X_offset, standardize);

        primal_value =
          model->primal(residual, Xbeta, beta, y, lambda, screened_set);

        dual_scale = std::max(lambda, max(abs(c(screened_set))));
        theta = residual / dual_scale;
        dual_value = model->dual(theta, y, lambda);
        duality_gap = primal_value - dual_value;

        duality_gap_rel = duality_gap / std::max(GAP_EPS, primal_value);

        if (verbosity >= 2) {
          Rprintf("      global primal: %f, global dual: %f, global gap: %f\n",
                  primal_value,
                  dual_value,
                  duality_gap_rel);
        }

        if (duality_gap_rel <= tol_gap_rel)
          break;

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

          if (celer_use_old_dual) {
            // check if dual point from previous check performs better
            if (it > 0 && dual_value_old > dual_value) {
              if (verbosity >= 2)
                Rprintf("      using previous dual point\n");

              dual_value = dual_value_old;
              dual_scale = dual_scale_old;
              c = c_old;
            }

            // save dual objects for next iteration
            dual_value_old = dual_value;
            dual_scale_old = dual_scale;
            c_old = c;
          }

          duality_gap = primal_value - dual_value;
          duality_gap_rel = duality_gap / std::max(GAP_EPS, primal_value);

          if (verbosity >= 2) {
            Rprintf(
              "      global primal: %f, global dual: %f, global gap: %f\n",
              primal_value,
              dual_value,
              duality_gap_rel);
          }

          if (duality_gap_rel <= tol_gap_rel)
            break;

          vec d(p);
          d.fill(datum::inf);

          d(screened_set) = (1.0 - abs(c(screened_set)) / dual_scale) /
                            sqrt(X_norms_squared(screened_set));

          if (celer_prune) {
            tol_gap_rel_inner = duality_gap * 0.3;
            uvec active_set = find(beta != 0);

            d(active_set).fill(-1);

            if (it > 0) {
              ws_size = std::min(2 * active_set.n_elem, p);
            }
          } else {
            d(working_set).fill(-1);

            if (it > 0) {
              ws_size = std::min(2 * ws_size, p);
            }
          }

          if (verbosity >= 2) {
            Rprintf("      n_active: %i, n_working: %i, n_screened: %i\n",
                    active_set.n_elem,
                    ws_size,
                    screened_set.n_elem);
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
        }
      }

      if (screening_type == "blitz") {
        if (inner_solver_converged) {
          primal_value =
            model->primal(residual, Xbeta, beta, y, lambda, screened_set);

          updateCorrelation(
            c, residual, X, screened_set, X_offset, standardize);

          dual_scale = std::max(lambda, max(abs(c(screened_set))));
          theta = residual / dual_scale;
          dual_value = model->dual(theta, y, lambda);

          duality_gap = primal_value - dual_value;
          duality_gap_rel = duality_gap / std::max(GAP_EPS, primal_value);

          if (verbosity >= 2) {
            Rprintf(
              "      global primal: %f, global dual: %f, global gap: %f\n",
              primal_value,
              dual_value,
              duality_gap_rel);
          }

          if (duality_gap_rel <= tol_gap_rel)
            break;

          vec d(p);
          d.fill(datum::inf);

          d(screened_set) = (1.0 - abs(c(screened_set)) / dual_scale) /
                            sqrt(X_norms_squared(screened_set));

          tol_gap_rel_inner = duality_gap * 0.3;

          uvec active_set = find(beta != 0);

          d(active_set).fill(-1);

          if (it > 0) {
            ws_size = std::min(2 * active_set.n_elem, p);

            // TODO(jolars): The commented-out code below is actually
            // what blitz does, but it's not efficient for low p or n.
            // if (ws_size < 100)
            //   ws_size = 100;
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

          if (verbosity >= 2) {
            Rprintf("      n_active: %i, n_working: %i, n_screened: %i\n",
                    active_set.n_elem,
                    ws_size,
                    screened_set.n_elem);
          }

          ws_size = std::min(ws_size, screened_set.n_elem);
          uvec ind = sort_index(d(screened_set), "ascend");
          working_set = screened_set(ind.head(ws_size));
        }
      }

      n_screened += screened_set.n_elem;

      if (line_search) {
        // this code is based on https://github.com/tbjohns/BlitzL1 as of
        // 2022-01-12, which is licensed under the MIT license, Copyright
        // Tyler B. Johnson 2015
        ws_size = working_set.n_elem;

        vec X_delta_beta(n, fill::zeros);
        vec delta_beta(ws_size, fill::zeros);
        vec hess_cache(ws_size);
        vec prox_newton_grad_cache(ws_size);

        double prox_newton_epsilon = 0;

        uword max_cd_itr = MAX_PROX_NEWTON_CD_ITR;

        w = model->weights(residual, y);

        for (uword j = 0; j < ws_size; ++j) {
          uword ind = working_set(j);
          hess_cache(j) = model->hessianTerm(X, ind, X_offset, standardize);
        }

        if (first_prox_newton_iteration) {
          max_cd_itr = MIN_PROX_NEWTON_CD_ITR;
          first_prox_newton_iteration = false;
          prox_newton_grad_diff = 0;

          updateCorrelation(c, residual, X, working_set, X_offset, standardize);
          prox_newton_grad_cache = -c(working_set);
        } else {
          prox_newton_epsilon =
            PROX_NEWTON_EPSILON_RATIO * prox_newton_grad_diff;
        }

        for (uword it_inner = 0; it_inner < max_cd_itr; ++it_inner) {

          if (shuffle || !progress) {
            uvec perm = randperm(ws_size);

            working_set = working_set(perm);
            delta_beta = delta_beta(perm);
            prox_newton_grad_cache = prox_newton_grad_cache(perm);
            hess_cache = hess_cache(perm);
          }

          double sum_sq_hess_diff = 0;

          for (uword j = 0; j < ws_size; ++j) {
            uword ind = working_set(j);
            double hess_j = hess_cache(j);

            double grad = prox_newton_grad_cache(j) +
                          weightedInnerProduct(
                            X, ind, X_delta_beta, w, X_offset, standardize);

            double old_value = beta(ind) + delta_beta(j);
            double proposal = old_value - grad / hess_j;
            double new_value = prox(proposal, lambda / hess_j);
            double diff = new_value - old_value;

            if (diff != 0) {
              delta_beta(j) = new_value - beta(ind);

              // X_delta_beta += X.col(j) * diff;
              addScaledColumn(
                X_delta_beta, X, ind, diff, X_offset, standardize);
              sum_sq_hess_diff += diff * diff * hess_j * hess_j;
            }
          }

          if (sum_sq_hess_diff < prox_newton_epsilon &&
              it_inner + 1 >= MIN_PROX_NEWTON_CD_ITR) {
            break;
          }
        }

        double t = 1;
        double last_t = 0;

        for (uword backtrack_itr = 0; backtrack_itr < MAX_BACKTRACK_ITR;
             ++backtrack_itr) {
          double diff_t = t - last_t;

          double subgrad_t = 0;

          for (uword j = 0; j < ws_size; ++j) {
            uword ind = working_set(j);
            beta(ind) += diff_t * delta_beta(j);

            if (beta(ind) < 0)
              subgrad_t -= lambda * delta_beta(j);
            else if (beta(ind) > 0)
              subgrad_t += lambda * delta_beta(j);
            else
              subgrad_t -= lambda * std::abs(delta_beta(j));
          }

          Xbeta += diff_t * X_delta_beta;

          model->updateResidual(residual, Xbeta, y);

          subgrad_t += dot(X_delta_beta, -residual);

          if (subgrad_t < 0) {
            break;
          } else {
            last_t = t;
            t *= 0.5;
          }
        }

        // cache gradients for next iteration
        if (t != 1) {
          X_delta_beta *= t;
        }

        updateCorrelation(c, residual, X, working_set, X_offset, standardize);

        for (uword j = 0; j < ws_size; ++j) {
          uword ind = working_set(j);

          double actual_grad = -c(ind);
          double approximate_grad =
            prox_newton_grad_cache(j) +
            weightedInnerProduct(X, ind, X_delta_beta, w, X_offset, standardize);

          prox_newton_grad_cache(j) = actual_grad;
          double diff = actual_grad - approximate_grad;
          prox_newton_grad_diff += diff * diff;
        }
      } else {
        if (shuffle || !progress)
          working_set = arma::shuffle(working_set);

        for (auto&& j : working_set) {
          updateCorrelation(c, residual, X, j, X_offset, standardize);
          double hess_j = model->hessianTerm(X, j, X_offset, standardize);
          double beta_j_old = beta(j);
          double v =
            prox(beta_j_old + c(j) / hess_j, lambda / hess_j) - beta(j);

          if (v != 0) {
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

      if (screening_type == "celer" && celer_use_accel) {
        U = join_horiz(U.tail_cols(K - 1), residual - residual_prev);

        residual_storage =
          join_horiz(residual_storage.tail_cols(K - 1), residual);
        residual_prev = residual;
      }

      it++;

      if (screening_type != "gap_safe" && it % CHECK_FREQUENCY == 0) {
        primal_value =
          model->primal(residual, Xbeta, beta, y, lambda, working_set);

        if (line_search) {
          // correlation vector is always updated at end of line search
          updateCorrelation(c, residual, X, working_set, X_offset, standardize);
        }

        dual_scale = std::max(lambda, max(abs(c(working_set))));
        theta = residual / dual_scale;
        dual_value = model->dual(theta, y, lambda);

        duality_gap = primal_value - dual_value;
        duality_gap_rel = duality_gap / std::max(GAP_EPS, primal_value);

        if (screening_type == "celer" && celer_use_accel && it >= K) {
          // use dual extrapolation
          bool success =
            solve(z, symmatu(U.t() * U), ones<vec>(K), solve_opts::no_approx);

          if (success) {
            // if solver succeeds (well-conditioned problem), use acceleration
            celer_c = z / accu(z);

            vec residual_accel(n, fill::zeros);

            for (uword i = 0; i < K; ++i) {
              residual_accel += celer_c(i) * residual_storage.col(K - i - 1);
            }

            vec c_accel(p, fill::zeros);

            updateCorrelation(
              c_accel, residual_accel, X, working_set, X_offset, standardize);

            double dual_scale_accel =
              std::max(lambda, max(abs(c_accel(working_set))));
            vec theta_accel = residual_accel / dual_scale_accel;
            double dual_value_accel = model->dual(theta_accel, y, lambda);

            if (dual_value_accel > dual_value) {
              dual_value = dual_value_accel;
              theta = theta_accel;
            }
          }
        }

        if (verbosity >= 2)
          Rprintf("      primal: %f, dual: %f, gap: %f\n",
                  primal_value,
                  dual_value,
                  duality_gap_rel);

        inner_solver_converged = duality_gap_rel <= tol_gap_rel_inner;

        if (line_search) {
          if (primal_value >= primal_value_prev)
            inner_solver_converged = true;

          primal_value_prev = primal_value;
          first_prox_newton_iteration = true;
        }

        if (inner_solver_converged && screening_type != "celer" &&
            screening_type != "blitz")
          break;

        progress = duality_gap_rel < duality_gap_rel_prev;
        duality_gap_rel_prev = duality_gap_rel;

        if (!progress && verbosity >= 2)
          Rprintf("      no progress; shuffling indices\n");

        duality_gap_rel_prev = duality_gap_rel;
      }

      if (it % 10 == 0) {
        Rcpp::checkUserInterrupt();
      }
    }
  } else {
    beta.zeros();
  }

  double avg_screened = n_screened / (it + 1);

  return { primal_value, dual_value, duality_gap, it + 1, avg_screened };
}

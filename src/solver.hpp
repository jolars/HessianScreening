#pragma once

#include "kktCheck.h"
#include "linearAlgebra.h"
#include "model.h"
#include "safeScreening.h"
#include "sasviScreening.h"
#include "updateCorrelation.h"
#include "updateLinearPredictor.h"
#include "utils.h"
#include <RcppArmadillo.h>

template<typename T>
std::tuple<double,
           double,
           double,
           arma::vec,
           arma::uword,
           double,
           arma::uword,
           arma::uword,
           double,
           double>
fit(arma::uvec& screened,
    arma::vec& c,
    arma::vec& residual,
    arma::vec& Xbeta,
    arma::vec& beta,
    const std::unique_ptr<Model>& model,
    const T& X,
    const arma::vec& y,
    const arma::vec& X_norms,
    const arma::vec& X_offset,
    const double y_norm,
    const arma::vec& XTy,
    const bool standardize,
    const arma::uvec& active_set,
    const arma::uvec& strong_set,
    const double lambda,
    const double lambda_prev,
    const double lambda_max,
    const arma::uword n_active_prev,
    const std::string screening_type,
    const bool shuffle,
    const arma::uword check_frequency,
    const arma::uword screen_frequency,
    const bool celer_use_old_dual,
    const bool celer_use_accel,
    const bool celer_prune,
    const bool gap_safe_active_start,
    const bool augment_with_gap_safe,
    const arma::uword step,
    const arma::uword maxit,
    const double tol_gap,
    const bool line_search,
    const arma::uword ws_size_init,
    const arma::uword verbosity)
{
  using namespace arma;

  const uword n = X.n_rows;
  const uword p = X.n_cols;

  if (screening_type == "celer" || screening_type == "gap_safe" ||
      screening_type == "blitz" || screening_type == "sasvi" ||
      screening_type == "none")
    screened.fill(true);

  uvec screened_set = find(screened);
  uvec working_set = screened_set;
  uvec safe(p, fill::ones);
  uvec safe_set = find(safe);
  uvec violations(p);

  uword n_refits{ 0 };
  uword n_violations{ 0 };

  double primal_value =
    model->primal(residual, Xbeta, beta, y, lambda, screened_set);
  double primal_value_prev = primal_value;

  double dual_scale = std::max(lambda, max(abs(c)));
  vec theta = residual / dual_scale;
  double dual_value = model->dual(theta, y, dual_scale, lambda);
  double dual_value_prev = dual_value;

  vec w(n, fill::ones);

  double duality_gap = primal_value - dual_value;

  bool inner_solver_converged = false;
  bool progress = true;
  double dual_scale_old = dual_scale;
  double dual_value_old = dual_value;

  uword ws_size = ws_size_init;
  vec c_old(p);
  vec d(p);
  vec residual_old;

  double tol_mod = model->toleranceModifier(y);

  // celer acceleration
  const uword K = 5;
  mat U(n, K);
  mat residual_storage(n, K);
  vec celer_c(K);
  vec residual_prev;
  vec z(K);

  // gap-safe stuff
  bool active_start = gap_safe_active_start;

  // line search type 1 paramters
  const uword MAX_BACKTRACK_ITR = 20;
  const uword MAX_PROX_NEWTON_CD_ITR = 20;
  const double PROX_NEWTON_EPSILON_RATIO = 10;
  const uword MIN_PROX_NEWTON_CD_ITR = 2;
  double prox_newton_grad_diff = 0;

  if (screening_type == "celer") {
    residual_old = residual;
    residual_prev = residual;
  }

  double n_screened = 0;
  uword it = 0;
  uword it_inner = 0;
  double tol_gap_inner = tol_gap * tol_mod;

  // timing
  wall_clock timer;
  timer.tic();

  double cd_time = 0;
  double kkt_time = 0;

  if (!screened_set.is_empty()) {
    updateLinearPredictor(Xbeta, X, beta, X_offset, standardize, active_set);
    model->updateResidual(residual, Xbeta, y);

    while (it < maxit) {
      if (verbosity >= 2) {
        Rprintf("    iter: %i\n", it + 1);
      }

      if (screening_type == "none" && inner_solver_converged) {
        // updateCorrelation(c, residual, X, X_offset, standardize);
        break;
      }

      if (screening_type == "hessian" || screening_type == "edpp" ||
          screening_type == "working" || screening_type == "strong") {
        if (inner_solver_converged && it > 0) {
          double t0 = timer.toc();

          bool outer_check = false;

          uvec unscreened_set = find(safe && (screened == false));

          violations.fill(false);

          uvec check_set;

          if (screening_type != "strong" && screening_type != "edpp") {
            check_set = setIntersect(unscreened_set, strong_set);
            updateCorrelation(c, residual, X, check_set, X_offset, standardize);
            kktCheck(violations, screened, c, check_set, lambda);
          }

          if (!any(violations)) {
            if (screening_type == "working" || screening_type == "hessian") {
              check_set = check_set = setDiff(unscreened_set, strong_set);
            } else {
              check_set = unscreened_set;
            }
            updateCorrelation(c, residual, X, check_set, X_offset, standardize);
            kktCheck(violations, screened, c, check_set, lambda);

            outer_check = true;
          }

          screened_set = find(screened);

          kkt_time += timer.toc() - t0;

          primal_value =
            model->primal(residual, Xbeta, beta, y, lambda, working_set);

          dual_scale = std::max(lambda, max(abs(c(safe_set))));
          theta = residual / dual_scale;
          dual_value = model->dual(theta, y, dual_scale, lambda);
          duality_gap = primal_value - dual_value;

          if (verbosity >= 2) {
            Rprintf("    GLOBAL primal: %f, dual: %f, gap: %f, tol: %f\n",
                    primal_value,
                    dual_value,
                    duality_gap,
                    tol_gap * tol_mod);
          }

          if (duality_gap <= tol_gap * tol_mod)
            break;

          if (any(violations)) {
            n_refits += 1;
          }

          if (outer_check && augment_with_gap_safe) {
            // have updated entire correlation vector, so let's use
            // it to gap-safe screen
            double r_screen = model->safeScreeningRadius(duality_gap, lambda);

            safeScreening(safe,
                          safe_set,
                          residual,
                          Xbeta,
                          beta,
                          c,
                          dual_scale,
                          r_screen,
                          model,
                          X,
                          y,
                          X_offset,
                          standardize,
                          X_norms);

            screened = screened && safe;
            screened_set = find(screened);
          }

          working_set = screened_set;

          n_violations += sum(violations);
        }
      }

      if (screening_type == "gap_safe") {
        if (it == 0 && gap_safe_active_start) {
          working_set = active_set;
        } else if ((it % screen_frequency == 0) && !active_start) {
          double t0 = timer.toc();
          updateCorrelation(
            c, residual, X, screened_set, X_offset, standardize);

          primal_value =
            model->primal(residual, Xbeta, beta, y, lambda, screened_set);

          dual_scale = std::max(lambda, max(abs(c(screened_set))));
          theta = residual / dual_scale;
          dual_value = model->dual(theta, y, dual_scale, lambda);
          duality_gap = primal_value - dual_value;

          if (verbosity >= 2) {
            Rprintf("    GLOBAL primal: %f, dual: %f, gap: %f, tol: %f\n",
                    primal_value,
                    dual_value,
                    duality_gap,
                    tol_gap * tol_mod);
          }

          kkt_time += timer.toc() - t0;

          if (duality_gap <= tol_gap * tol_mod)
            break;

          double r_screen = model->safeScreeningRadius(duality_gap, lambda);

          safeScreening(screened,
                        screened_set,
                        residual,
                        Xbeta,
                        beta,
                        c,
                        dual_scale,
                        r_screen,
                        model,
                        X,
                        y,
                        X_offset,
                        standardize,
                        X_norms);

          working_set = screened_set;
        }
      }

      if (screening_type == "sasvi") {
        if (it == 0 && gap_safe_active_start) {
          working_set = active_set;
        } else if ((it % screen_frequency == 0) && !active_start) {
          double t0 = timer.toc();
          updateCorrelation(
            c, residual, X, screened_set, X_offset, standardize);

          primal_value =
            model->primal(residual, Xbeta, beta, y, lambda, screened_set);

          dual_scale = std::max(lambda, max(abs(c(screened_set))));
          theta = residual / dual_scale;
          dual_value = model->dual(theta, y, dual_scale, lambda);
          duality_gap = primal_value - dual_value;

          if (verbosity >= 2) {
            Rprintf("    GLOBAL primal: %f, dual: %f, gap: %f, tol: %f\n",
                    primal_value,
                    dual_value,
                    duality_gap,
                    tol_gap * tol_mod);
          }

          kkt_time += timer.toc() - t0;

          if (duality_gap <= tol_gap * tol_mod)
            break;

          // SASVI uses a different parametrization:
          // y := y/lambda and beta := beta/lambda
          double l1beta = norm(beta, 1) / lambda;
          vec XTXbeta = (XTy - c) / lambda;
          double l2Xbeta = norm(Xbeta) / lambda;
          double yTXbeta = dot(y, Xbeta) / (lambda * lambda);
          double k = lambda / max(abs(c(screened_set)));

          screening_dsasvi_lasso_lincombtheta(p,
                                              screened,
                                              XTXbeta,
                                              XTy / lambda,
                                              X_norms,
                                              l1beta,
                                              l2Xbeta,
                                              y_norm / lambda,
                                              yTXbeta,
                                              k,
                                              -k);

          for (uword j = 0; j < p; ++j) {
            if (beta(j) != 0 && !screened(j)) {
              model->adjustResidual(
                residual, Xbeta, X, y, j, -beta(j), X_offset, standardize);

              beta(j) = 0;
            }
          }

          screened_set = find(screened);
          working_set = screened_set;
        }
      }

      if (screening_type == "celer") {
        if (inner_solver_converged || it == 0) {
          primal_value =
            model->primal(residual, Xbeta, beta, y, lambda, screened_set);

          updateCorrelation(
            c, residual, X, screened_set, X_offset, standardize);

          dual_scale = std::max(lambda, max(abs(c(screened_set))));
          theta = residual / dual_scale;
          dual_value = model->dual(theta, y, dual_scale, lambda);

          bool celer_old_dual_better = false;

          vec c_check = c;

          if (celer_use_old_dual) {
            // check if dual point from previous check performs better
            celer_old_dual_better = it > 0 && dual_value_old > dual_value;

            if (celer_old_dual_better) {
              if (verbosity >= 2)
                Rprintf("      using previous dual point\n");

              dual_value = dual_value_old;
              dual_scale = dual_scale_old;
              c_check = c_old;
            }

            // save dual objects for next iteration
            dual_value_old = dual_value;
            dual_scale_old = dual_scale;
            c_old = c;
          }

          duality_gap = primal_value - dual_value;

          if (verbosity >= 2) {
            Rprintf("    GLOBAL primal: %f, dual: %f, gap: %f, tol: %f\n",
                    primal_value,
                    dual_value,
                    duality_gap,
                    tol_gap * tol_mod);
          }

          if (duality_gap <= tol_gap * tol_mod)
            break;

          vec d(p);
          d.fill(datum::inf);

          d(screened_set) =
            (1.0 - abs(c(screened_set)) / dual_scale) / X_norms(screened_set);

          if (celer_prune) {
            tol_gap_inner = duality_gap * 0.3;
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

          double r_screen = model->safeScreeningRadius(duality_gap, lambda);

          safeScreening(screened,
                        screened_set,
                        residual,
                        Xbeta,
                        beta,
                        c_check,
                        dual_scale,
                        r_screen,
                        model,
                        X,
                        y,
                        X_offset,
                        standardize,
                        X_norms);

          ws_size = std::min(ws_size, screened_set.n_elem);

          uvec ind = sort_index(d(screened_set), "ascend");

          working_set = screened_set(ind.head(ws_size));
        }
      }

      if (screening_type == "blitz") {
        if (inner_solver_converged || it == 0) {
          primal_value =
            model->primal(residual, Xbeta, beta, y, lambda, screened_set);

          updateCorrelation(
            c, residual, X, screened_set, X_offset, standardize);

          dual_scale = std::max(lambda, max(abs(c(screened_set))));
          theta = residual / dual_scale;
          dual_value = model->dual(theta, y, dual_scale, lambda);

          duality_gap = primal_value - dual_value;

          if (verbosity >= 2) {
            Rprintf("    GLOBAL primal: %f, dual: %f, gap: %f, tol: %f\n",
                    primal_value,
                    dual_value,
                    duality_gap,
                    tol_gap * tol_mod);
          }

          if (duality_gap <= tol_gap * tol_mod)
            break;

          vec d(p);
          d.fill(datum::inf);

          d(screened_set) =
            (1.0 - abs(c(screened_set)) / dual_scale) / X_norms(screened_set);

          tol_gap_inner = duality_gap * 0.3;

          uvec active_set = find(beta != 0);

          d(active_set).fill(-1);

          if (it > 0) {
            ws_size = std::min(2 * active_set.n_elem, p);

            // TODO(jolars): The commented-out code below is actually
            // what blitz does, but it's not efficient for low p or n.
            // if (ws_size < 100)
            //   ws_size = 100;
          }

          double r_screen = model->safeScreeningRadius(duality_gap, lambda);

          safeScreening(screened,
                        screened_set,
                        residual,
                        Xbeta,
                        beta,
                        c,
                        dual_scale,
                        r_screen,
                        model,
                        X,
                        y,
                        X_offset,
                        standardize,
                        X_norms);

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

      if (inner_solver_converged) {
        // reset inner solver variables
        it_inner = 0;
        inner_solver_converged = false;
        progress = true;
      }

      double t0 = timer.toc();

      if (!line_search) {
        // no line search
        if (it_inner != 0) {
          if (shuffle || !progress)
            working_set = arma::shuffle(working_set);
        }

        for (auto&& j : working_set) {
          updateCorrelation(c, residual, X, j, X_offset, standardize);
          double hess_j = model->hessianTerm(X, j, X_offset, standardize);

          if (hess_j <= 0)
            continue;

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
      } else {
        // blitz-type line search
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

        if (it_inner == 0) {
          max_cd_itr = MIN_PROX_NEWTON_CD_ITR;
          prox_newton_grad_diff = 0;

          updateCorrelation(c, residual, X, working_set, X_offset, standardize);
          prox_newton_grad_cache = -c(working_set);
        } else {
          prox_newton_epsilon =
            PROX_NEWTON_EPSILON_RATIO * prox_newton_grad_diff;
        }

        for (uword it_inner = 0; it_inner < max_cd_itr; ++it_inner) {

          // blitz always shuffles the working set
          uvec perm = randperm(ws_size);
          working_set = working_set(perm);
          delta_beta = delta_beta(perm);
          prox_newton_grad_cache = prox_newton_grad_cache(perm);
          hess_cache = hess_cache(perm);

          double sum_sq_hess_diff = 0;

          for (uword j = 0; j < ws_size; ++j) {
            uword ind = working_set(j);
            double hess_j = hess_cache(j);

            if (hess_j <= 0)
              continue;

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
            weightedInnerProduct(
              X, ind, X_delta_beta, w, X_offset, standardize);

          prox_newton_grad_cache(j) = actual_grad;
          double diff = actual_grad - approximate_grad;
          prox_newton_grad_diff += diff * diff;
        }
      }

      if (screening_type == "celer" && celer_use_accel && it_inner > 0) {
        sword ind = (it_inner - 1) % check_frequency - K;

        if (ind >= 0) {
          U.col(ind) = residual - residual_prev;
          residual_storage.col(ind) = residual;
        }
      }

      residual_prev = residual;

      bool check = screening_type == "gap_safe" || screening_type == "sasvi"
                     ? active_start
                     : true;

      if ((check && (it_inner % check_frequency == 0)) || line_search) {
        primal_value =
          model->primal(residual, Xbeta, beta, y, lambda, working_set);

        if (!line_search) {
          // correlation vector is always updated at end of line search
          updateCorrelation(c, residual, X, working_set, X_offset, standardize);
        }

        dual_scale = std::max(lambda, max(abs(c(working_set))));

        theta = residual / dual_scale;
        dual_value = model->dual(theta, y, dual_scale, lambda);

        duality_gap = primal_value - dual_value;

        if (screening_type == "celer" && celer_use_accel && it_inner >= K) {
          // use dual extrapolation
          bool success =
            solve(z, symmatu(U.t() * U), ones<vec>(K), solve_opts::no_approx);

          if (success) {
            // if solver succeeds (well-conditioned problem), use acceleration
            celer_c = z / accu(z);

            vec residual_accel(n, fill::zeros);

            for (uword i = 0; i < K; ++i)
              residual_accel += celer_c(i) * residual_storage.col(i);

            vec c_accel(p);

            updateCorrelation(
              c_accel, residual_accel, X, working_set, X_offset, standardize);

            double dual_scale_accel =
              std::max(lambda, max(abs(c_accel(working_set))));
            vec theta_accel = residual_accel / dual_scale_accel;
            double dual_value_accel =
              model->dual(theta_accel, y, dual_scale_accel, lambda);

            if (dual_value_accel > dual_value) {
              if (verbosity >= 2)
                Rprintf("      using accelerated dual point\n");

              dual_value = dual_value_accel;
              theta = theta_accel;
            }
          }
        }

        if (verbosity >= 2)
          Rprintf("      primal: %f, dual: %f, gap: %f, tol: %f\n",
                  primal_value,
                  dual_value,
                  duality_gap,
                  tol_gap_inner);

        inner_solver_converged = duality_gap <= tol_gap_inner;

        if (line_search) {
          // line search should ensure progress in the primal, so if primal
          // value increases, then the limit of machine precision must have
          // been reached
          if (primal_value >= primal_value_prev)
            inner_solver_converged = true;
        }

        progress =
          primal_value < primal_value_prev || dual_value > dual_value_prev;

        if (!progress && verbosity >= 2)
          Rprintf("      no progress; shuffling indices\n");

        if (inner_solver_converged) {
          if (verbosity >= 2)
            Rprintf("      inner solver converged\n");

          primal_value_prev = datum::inf;
          dual_value_prev = -datum::inf;

          active_start = false; // for gap safe
        } else {
          primal_value_prev = primal_value;
          dual_value_prev = dual_value;
        }
      }

      cd_time += timer.toc() - t0;

      if (it % 10 == 0) {
        Rcpp::checkUserInterrupt();
      }

      it++;
      it_inner++;
    }

  } else {
    beta.zeros();
  }

  // Return average number of screened predictors. If it == 0, then the
  // algorithm converged instantly and hence did not need to do any screening
  double avg_screened = it == 0 ? 0 : n_screened / it;

  return { primal_value, dual_value,   duality_gap, theta,   it,
           avg_screened, n_violations, n_refits,    cd_time, kkt_time };
}

#include <RcppArmadillo.h>

#include "setupModel.h"
#include "model.h"
#include "getNextLambda.h"
#include "colNormsSquared.h"
#include "binomial.h"
#include "gaussian.h"
#include "checkStoppingConditions.h"
#include "findDuplicates.h"
#include "kktCheck.h"
#include "rescaleCoefficients.h"
#include "screenPredictors.h"
#include "standardize.h"
#include "updateHessian.h"

using namespace arma;
using namespace Rcpp;

template<typename T>
Rcpp::List
lassoPathImpl(T X,
              arma::vec y,
              const std::string family,
              const bool X_is_sparse,
              const bool standardize,
              const std::string screening_type,
              const bool hessian_warm_starts,
              std::string log_hessian_update_type,
              const uword log_hessian_auto_threshold,
              const arma::uword path_length,
              const arma::uword maxit,
              const double tol_decr,
              const double tol_infeas,
              const double tol_gap,
              const double gamma,
              const bool verify_hessian,
              const bool force_kkt_check,
              const arma::uword verbosity)
{
  const uword n = X.n_rows;
  const uword p = X.n_cols;

  wall_clock timer;
  timer.tic();

  double full_time = timer.toc();

  const bool hessian_type_screening =
    screening_type == "hessian" || screening_type == "hessian_adaptive";

  vec beta(p, fill::zeros);
  mat betas(p, 0, fill::zeros);
  vec Xbeta(n, fill::zeros);
  vec residual(n, fill::zeros);
  vec c(p);
  vec c_pred(p);
  vec c_grad(p);

  // standardize predictors and response
  vec X_mean = zeros<vec>(p);
  vec X_sd = ones<vec>(p);

  if (standardize) {
    standardizeX(X_mean, X_sd, X);
  }

  const vec X_mean_scaled = X_mean / X_sd;
  const double y_center = mean(y);

  vec X_norms_squared(p);

  if (family == "gaussian" || screening_type == "gap_safe") {
    if (!standardize) {
      X_norms_squared = colNormsSquared(X);
    } else {
      X_norms_squared.fill(n);
    }
  }

  auto model = setupModel(family,
                          y,
                          beta,
                          residual,
                          Xbeta,
                          c,
                          X_mean_scaled,
                          X_norms_squared,
                          n,
                          p,
                          standardize,
                          log_hessian_update_type);

  model->standardizeY();

  model->updateResidual();
  model->updateCorrelation(X, regspace<uvec>(0, p - 1));

  const double lambda_min_ratio = n < p ? 0.01 : 1e-4;
  const double lambda_max = max(abs(c));
  const double lambda_min = lambda_max * lambda_min_ratio;

  const vec lambda_grid =
    exp(linspace(log(lambda_max), log(lambda_min), path_length));

  std::vector<double> lambdas;

  double lambda = lambda_max;

  const double lambda_min_step = 0.1 * min(abs(diff(lambda_grid)));
  double tmp = n < p ? n / path_length : p / path_length;

  uword n_target_nonzero = std::min(p, static_cast<uword>(std::ceil(tmp)));
  n_target_nonzero = std::max(static_cast<uword>(1), n_target_nonzero);

  GetNextLambda getNextLambda{ c,
                               c_grad,
                               lambda_grid,
                               lambda_min,
                               lambda_min_step,
                               screening_type,
                               n_target_nonzero,
                               verbosity };

  std::vector<double> primals;
  std::vector<double> duals;
  std::vector<double> devs;
  std::vector<double> dev_ratios;

  std::vector<uword> n_active;
  std::vector<uword> n_new_active;
  std::vector<uword> n_passes;
  std::vector<uword> n_refits;
  std::vector<uword> n_screened;
  std::vector<uword> n_strong;
  std::vector<uword> n_violations;

  uvec active(p, fill::zeros);
  std::vector<uword> originals;
  std::vector<uword> duplicates;
  uvec duplicated(p, fill::zeros);

  uvec first_active = find(abs(c) == lambda_max);

  for (uword i = 1; i < first_active.n_elem; ++i) {
    originals.emplace_back(first_active(0));
    duplicates.emplace_back(first_active(i));
  }

  active(first_active(0)) = true;

  uvec active_prev = active;
  uvec ever_active = active;
  uvec screened = active;
  uvec strong = active;
  uvec strong_set = find(strong);

  uvec violations(p, fill::zeros);

  uvec active_perm = find(active);
  uvec active_perm_prev = active_perm;
  uvec active_set = active_perm;
  uvec active_set_prev = active_set;

  mat H = model->hessian(X, active_perm);
  mat Hinv = inv(symmatl(H));

  vec s(p, fill::zeros);
  s(active_perm) = sign(c(active_perm));

  vec Hinv_s = Hinv * s(active_perm);

  const double null_dev = model->deviance();
  double dev = null_dev;

  bool check_kkt = (screening_type != "gap_safe" && screening_type != "edpp") ||
                   force_kkt_check;

  std::vector<double> it_times;
  std::vector<double> cd_times;
  std::vector<double> kkt_times;
  std::vector<double> gradcorr_times;
  std::vector<double> hess_times;
  std::vector<double> duplicates_times;

  uword i = 0;

  while (true) {
    i++;

    double it_time = timer.toc();

    if (verbosity >= 1) {
      Rprintf("step: %i, lambda: %.2f\n", i, lambda);
    }

    vec beta_prev = beta;
    double dev_prev = dev;
    uword n_passes_i_sum = 0;
    uword n_violations_i = 0;
    uword n_refits_i = 0;
    bool first_run = true;

    double cd_time = 0;
    double kkt_time = 0;

    while (true) {
      if (verbosity >= 1) {
        Rprintf("  running coordinate descent\n");
      }

      double t0 = timer.toc();

      auto screening_type_choice =
        first_run && screening_type == "gap_safe" ? "working" : screening_type;

      if (first_run && screening_type != "gap_safe") {
        n_screened.emplace_back(sum(screened));
      }

      auto [primal_value, dual_value, duality_gap, n_passes_i, avg_screened] =
        model->fit(screened,
                   X,
                   X_norms_squared,
                   lambda,
                   lambda_max,
                   null_dev,
                   screening_type_choice,
                   maxit,
                   tol_decr,
                   tol_gap,
                   tol_infeas,
                   verbosity);

      cd_time += timer.toc() - t0;

      n_passes_i_sum += n_passes_i;

      if (screening_type_choice == "gap_safe") {
        // For dynamic screening rules, `avg_screened` is the mean number of
        // screened predictors. For other rules, this is constant between
        // iterations.
        n_screened.emplace_back(avg_screened);
      }

      t0 = timer.toc();

      if (check_kkt) {
        uvec unscreened_set = find(screened == false && duplicated == false);

        violations.fill(false);

        if (screening_type == "strong") {
          model->updateCorrelation(X, unscreened_set);
          kktCheck(violations, screened, c, unscreened_set, lambda);

        } else {
          uvec check_set = setIntersect(unscreened_set, strong_set);
          model->updateCorrelation(X, check_set);
          kktCheck(violations, screened, c, check_set, lambda);

          if (!any(violations)) {
            uvec check_set = setDiff(unscreened_set, strong_set);
            model->updateCorrelation(X, check_set);
            kktCheck(violations, screened, c, check_set, lambda);
          }
        }
      }

      kkt_time += timer.toc() - t0;

      n_violations_i += sum(violations);

      if (!any(violations) && !(screening_type == "gap_safe" && first_run)) {
        duals.emplace_back(dual_value);
        primals.emplace_back(primal_value);
        n_passes.emplace_back(n_passes_i_sum);
        n_refits.emplace_back(n_refits_i);
        n_violations.emplace_back(n_violations_i);
        lambdas.emplace_back(lambda);
        cd_times.emplace_back(cd_time);
        kkt_times.emplace_back(kkt_time);

        break;
      } else {
        n_refits_i++;
      }

      first_run = false;

      Rcpp::checkUserInterrupt();
    }

    if (i > 1) {
      active = beta != 0;
      active_set = find(active);
      s.zeros();
      s(active_set) = sign(c(active_set));
    }

    active_perm = join_vert(safeSetIntersect(active_perm_prev, active_set),
                            setDiff(active_set, active_set_prev));

    dev = model->deviance();
    devs.emplace_back(dev);
    dev_ratios.emplace_back(1.0 - dev / null_dev);

    // find duplicates among the just-activated predictors, drop them, and
    // adjust the coefficients accordingly
    double t0 = timer.toc();

    auto [new_originals, new_duplicates] =
      findDuplicates(active_set, active_set_prev, X, model);

    if (!new_duplicates.empty()) {
      for (auto&& orig : new_originals) {
        uvec dups = new_duplicates(find(new_originals == orig));

        beta(orig) = signum(c(orig)) * sum(abs(beta(dups)));
        beta(dups).zeros();
        s(dups).zeros();
      }

      originals.insert(
        originals.end(), new_originals.begin(), new_originals.end());
      duplicates.insert(
        duplicates.end(), new_duplicates.begin(), new_duplicates.end());

      duplicated(new_duplicates).fill(true);
      active(new_duplicates).fill(false);
      active_perm = safeSetDiff(active_perm, new_duplicates);
      active_set = find(active);
      ever_active(new_duplicates).fill(false);
    }

    duplicates_times.emplace_back(timer.toc() - t0);

    uword new_active = setDiff(active_set, active_set_prev).n_elem;
    ever_active(active_set).fill(true);
    n_active.emplace_back(active_set.n_elem);
    n_new_active.emplace_back(new_active);

    betas.insert_cols(betas.n_cols, beta);

    if (active_set.n_elem > log_hessian_auto_threshold &&
        log_hessian_update_type == "auto") {

      log_hessian_update_type = "approx";
      model->setLogHessianUpdateType("approx");
      active_perm = active_set;
      active_perm_prev = active_set_prev;
    }

    if (verbosity >= 1) {
      Rprintf("  active: %i, new active: %i\n", active_set.n_elem, new_active);
    }

    bool stop_path = checkStoppingConditions(i,
                                             n,
                                             p,
                                             path_length,
                                             active_set.n_elem,
                                             lambda,
                                             lambda_min,
                                             dev,
                                             dev_prev,
                                             null_dev,
                                             screening_type,
                                             verbosity);

    if (stop_path) {
      hess_times.emplace_back(0);
      it_times.emplace_back(timer.toc() - it_time);
      break;
    }

    bool reset_hessian = false;

    if (hessian_type_screening) {
      double t0 = timer.toc();

      if (log_hessian_update_type == "approx" || family == "gaussian") {
        updateHessian(H,
                      Hinv,
                      active_set,
                      active_set_prev,
                      active_perm,
                      active_perm_prev,
                      model,
                      X,
                      verify_hessian,
                      verbosity,
                      reset_hessian);

        Hinv_s = Hinv * s(active_perm);
        Hinv_s = Hinv_s(sort_index(active_perm)); // reset permutation
      } else {
        // for logistic regression and no approxiation, simply recompute the
        // hessian and its inverse for the full set of active predictors,
        // since we cannot update the hessian efficiently anyway
        H = model->hessian(X, active_set);

        vec eigval;
        mat eigvec;

        eig_sym(eigval, eigvec, symmatu(H));

        if (eigval.min() < 1e-4 * n) {
          H.diag() += 1e-4 * n;
          eigval += 1e-4 * n;
        }

        Hinv = eigvec * diagmat(1 / eigval) * eigvec.t();
        Hinv_s = Hinv * s(active_set);
      }

      hess_times.emplace_back(timer.toc() - t0);

      // for hessian_adaptive we need to use all predictors, but this is not the
      // case for the standard hessian method
      uvec restricted = screening_type == "hessian"
                          ? abs(c) >= 2 * lambda_grid(i) - lambda
                          : abs(c) >= 2 * lambda_min - lambda;

      t0 = timer.toc();

      model->updateGradientOfCorrelation(
        c_grad, X, Hinv_s, s, active_set, find(restricted));

      gradcorr_times.emplace_back(timer.toc() - t0);
    }

    double lambda_next = getNextLambda(lambda, active, new_active, i);

    strong = abs(c) >= 2 * lambda_next - lambda;
    strong_set = find(strong);
    n_strong.emplace_back(sum(strong));

    screened = screenPredictors(model,
                                screening_type,
                                strong,
                                ever_active,
                                residual,
                                c,
                                c_grad,
                                X,
                                X_norms_squared,
                                X_mean_scaled,
                                y,
                                lambda,
                                lambda_next,
                                gamma,
                                X_is_sparse,
                                standardize);

    // make sure duplicates stay out
    screened(find(duplicated)).fill(false);

    if (hessian_warm_starts && hessian_type_screening) {
      beta(active_set) += (lambda - lambda_next) * Hinv_s;
    }

    active_perm_prev = active_perm;
    active_set_prev = active_set;

    lambda = lambda_next;

    it_times.emplace_back(timer.toc() - it_time);

    Rcpp::checkUserInterrupt();
  }

  rescaleCoefficients(betas, X_mean, X_sd, y_center);

  full_time = timer.toc() - full_time;

  umat duplicates_mat = join_horiz(uvec(originals), uvec(duplicates));

  return List::create(Named("beta") = wrap(betas),
                      Named("lambda") = wrap(lambdas),
                      Named("primals") = wrap(primals),
                      Named("duals") = wrap(duals),
                      Named("dev_ratio") = wrap(dev_ratios),
                      Named("dev") = wrap(devs),
                      Named("violations") = wrap(n_violations),
                      Named("refits") = wrap(n_refits),
                      Named("active") = wrap(n_active),
                      Named("screened") = wrap(n_screened),
                      Named("strong") = wrap(n_strong),
                      Named("new_active") = wrap(n_new_active),
                      Named("duplicates") = wrap(duplicates_mat),
                      Named("passes") = wrap(n_passes),
                      Named("full_time") = wrap(full_time),
                      Named("it_time") = wrap(it_times),
                      Named("cd_time") = wrap(cd_times),
                      Named("hess_time") = wrap(hess_times),
                      Named("kkt_time") = wrap(kkt_times),
                      Named("gradcorr_time") = wrap(gradcorr_times));
}

//' Fit the Lasso Path
//'
//' @param X The predictor matrix
//' @param y The reponse vector
//' @param family The name of the family, "gaussian" or "logistic"
//' @param standardize Whether to standardize the predictors
//' @param screening_type Which screening type to use, currently
//'        `"hessian"`, `"working"`,`"gap_safe"`, or `"edpp"`.
//' @param hessian_warm_starts Whether to use warm starts based on Hessian
//' @param log_hessian_update_type what type of strategy to use for
//'        updating the hessian for logistic regression
//' @param log_hessian_auto_threshold if `log_hessian_update_type == "auto"`,
// this '        number decides when the updates switch from the full hessian
// update '        to the approximation ' @param path_length The length of the
// lasso path ' @param maxit Maximum number of iterations for Coordinate Descent
// loop ' @param tol_decr Tolerance threshold for change in primal value '
// @param tol_infeas Tolerance threshold for maximum infeasibility ' @param
// tol_gap Tolerance threshold for duality gap ' @param gamma Percent of strong
// approximation to add to Hessian '        approximation ' @param
// verify_hessian Whether ot not to verify that Hessian '        updates are
// correct. Used only for diagnostic purposes. ' @param verbosity Controls the
// level of verbosity. 0 = no output. ' @export
// [[Rcpp::export]]
Rcpp::List
lassoPath(SEXP X,
          arma::vec y,
          const std::string family = "gaussian",
          const bool standardize = true,
          const std::string screening_type = "working",
          const bool hessian_warm_starts = true,
          std::string log_hessian_update_type = "auto",
          const arma::uword log_hessian_auto_threshold = 500,
          const arma::uword path_length = 100,
          const arma::uword maxit = 1e5,
          const double tol_decr = 1e-7,
          const double tol_infeas = 1e-5,
          const double tol_gap = 1e-4,
          const double gamma = 0.01,
          const bool verify_hessian = false,
          const bool force_kkt_check = false,
          const arma::uword verbosity = 0)
{
  if (Rf_isS4(X)) {
    if (Rf_inherits(X, "dgCMatrix")) {
      return lassoPathImpl(as<arma::sp_mat>(X),
                           y,
                           family,
                           true,
                           standardize,
                           screening_type,
                           hessian_warm_starts,
                           log_hessian_update_type,
                           log_hessian_auto_threshold,
                           path_length,
                           maxit,
                           tol_decr,
                           tol_infeas,
                           tol_gap,
                           gamma,
                           verify_hessian,
                           force_kkt_check,
                           verbosity);
    }
  } else {
    return lassoPathImpl(as<arma::mat>(X),
                         y,
                         family,
                         false,
                         standardize,
                         screening_type,
                         hessian_warm_starts,
                         log_hessian_update_type,
                         log_hessian_auto_threshold,
                         path_length,
                         maxit,
                         tol_decr,
                         tol_infeas,
                         tol_gap,
                         gamma,
                         verify_hessian,
                         force_kkt_check,
                         verbosity);
  }

  // should never end up here
  return Rcpp::List::create();
}

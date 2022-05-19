#include "binomial.h"
#include "findDuplicates.h"
#include "gaussian.h"
#include "kktCheck.h"
#include "model.h"
#include "rescaleCoefficients.h"
#include "safeScreening.h"
#include "screenPredictors.h"
#include "setupModel.h"
#include "solver.hpp"
#include "squaredColNorms.h"
#include "standardize.h"
#include "updateCorrelation.h"
#include "updateHessian.h"
#include "updateLinearPredictor.h"
#include <RcppArmadillo.h>

template<typename T>
Rcpp::List
lassoPath(T& X,
          arma::vec& y,
          const std::string family,
          arma::vec lambdas,
          const std::string lambda_type,
          const bool standardize,
          const std::string screening_type,
          const bool shuffle,
          const arma::uword check_frequency,
          const arma::uword screen_frequency,
          const bool hessian_warm_starts,
          const bool celer_use_old_dual,
          const bool celer_use_accel,
          const bool celer_prune,
          const bool gap_safe_active_start,
          const bool augment_with_gap_safe,
          std::string log_hessian_update_type,
          const arma::uword log_hessian_auto_update_freq,
          arma::uword path_length,
          const arma::uword maxit,
          const double tol_gap,
          const double gamma,
          const bool store_dual_variables,
          const bool verify_hessian,
          const bool line_search,
          const arma::uword verbosity)
{
  using namespace arma;
  using namespace Rcpp;

  const uword n = X.n_rows;
  const uword p = X.n_cols;

  if (family == "binomial") {
    vec y_unique = sort(unique(y));

    if (y_unique.n_elem != 2) {
      Rcpp::stop("y has more than two unique values");
    } else if (y_unique(0) != 0 || y_unique(1) != 1) {
      Rcpp::stop("y is not in {0, 1}");
    }

    if (screening_type == "edpp" || screening_type == "sasvi")
      Rcpp::stop("EDPP and SASVI cannot be used in logistic regression");
  }

  bool log_hessian_auto = log_hessian_update_type == "auto";

  if (log_hessian_auto) {
    log_hessian_update_type = "full";
  }

  vec beta(p, fill::zeros);
  mat betas(p, 0, fill::zeros);
  mat thetas(n, 0);
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

  const vec X_offset = X_mean / X_sd;
  const double y_center = mean(y);
  const double y_norm = norm(y);

  const vec X_norms_squared = squaredColNorms(X, X_offset, standardize);
  const vec X_norms = sqrt(X_norms_squared);

  const uword ws_size_init = std::min(p, static_cast<uword>(100));

  auto model = setupModel(family, X_norms_squared, n, log_hessian_update_type);

  model->standardizeY(y);

  model->updateResidual(residual, Xbeta, y);
  updateCorrelation(c, residual, X, X_offset, standardize);

  const double lambda_min_ratio = n < p ? 1e-2 : 1e-4;
  const double lambda_max = max(abs(c));
  const double lambda_min = lambda_max * lambda_min_ratio;

  if (lambda_type == "auto") {
    lambdas = exp(linspace(log(lambda_max), log(lambda_min), path_length));
  } else {
    path_length = lambdas.n_elem;
  }

  std::vector<double> lambda_out;

  double lambda_prev = 2 * lambda_max;
  double lambda = lambda_max;

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

  uvec active_perm = find(active);
  uvec active_perm_prev = active_perm;
  uvec active_set = active_perm;
  uvec active_set_prev = active_set;

  mat H = model->hessian(X, active_perm, X_offset, standardize);
  mat Hinv = inv(symmatl(H));

  vec s(p, fill::zeros);
  s(active_perm) = sign(c(active_perm));

  vec Hinv_s = Hinv * s(active_perm);

  vec XTy(p); // only used with SASVI screening rule

  if (screening_type == "sasvi") {
    for (uword j = 0; j < p; ++j)
      XTy(j) = innerProduct(X, j, y, X_offset, standardize);
  }

  const double null_dev = model->deviance(residual, Xbeta, y);
  double dev = null_dev;
  double dev_prev = dev;

  std::string screening_type_temp = screening_type;

  std::vector<double> cd_time;
  std::vector<double> duplicates_time;
  std::vector<double> gradcorr_time;
  std::vector<double> hess_time;
  std::vector<double> it_time;
  std::vector<double> kkt_time;

  std::vector<bool> converged;

  wall_clock timer;
  timer.tic();

  double full_time = timer.toc();

  uword i = 0;

  while (true) {
    i++;

    double it_time_i = timer.toc();

    if (verbosity >= 1) {
      Rprintf("step: %i, lambda: %.2f\n", i, lambda);
    }

    vec beta_prev = beta;

    if (verbosity >= 1) {
      Rprintf("  running coordinate descent\n");
    }

    auto [primal_value,
          dual_value,
          duality_gap,
          theta,
          n_passes_i,
          avg_screened,
          n_violations_i,
          n_refits_i,
          cd_time_i,
          kkt_time_i] = fit(screened,
                            c,
                            residual,
                            Xbeta,
                            beta,
                            model,
                            X,
                            y,
                            X_norms,
                            X_offset,
                            y_norm,
                            XTy,
                            standardize,
                            active_set_prev,
                            strong_set,
                            lambda,
                            lambda_prev,
                            lambda_max,
                            active_set.n_elem,
                            screening_type_temp,
                            shuffle,
                            check_frequency,
                            screen_frequency,
                            celer_use_old_dual,
                            celer_use_accel,
                            celer_prune,
                            gap_safe_active_start,
                            augment_with_gap_safe,
                            i,
                            maxit,
                            tol_gap,
                            line_search,
                            ws_size_init,
                            verbosity);

    if (n_passes_i >= maxit) {
      converged.emplace_back(false);
    } else {
      converged.emplace_back(true);
    }

    if (store_dual_variables) {
      thetas.insert_cols(thetas.n_cols, theta);
    }

    duals.emplace_back(dual_value);
    primals.emplace_back(primal_value);
    n_passes.emplace_back(n_passes_i);
    n_refits.emplace_back(n_refits_i);
    n_violations.emplace_back(n_violations_i);
    lambda_out.emplace_back(lambda);
    cd_time.emplace_back(cd_time_i);
    kkt_time.emplace_back(kkt_time_i);
    n_screened.emplace_back(avg_screened);

    if (i > 1) {
      active = beta != 0;
      active_set = find(active);
      s.zeros();
      s(active_set) = sign(c(active_set));
    }

    active_perm = join_vert(safeSetIntersect(active_perm_prev, active_set),
                            setDiff(active_set, active_set_prev));

    dev_prev = dev;
    dev = model->deviance(residual, Xbeta, y);
    devs.emplace_back(dev);
    dev_ratios.emplace_back(1.0 - dev / null_dev);

    double t0 = timer.toc();

    // find duplicates among the just-activated predictors, drop them, and
    // adjust the coefficients accordingly
    auto [new_originals, new_duplicates] = findDuplicates(
      active_set, active_set_prev, X, model, X_offset, standardize);

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

    duplicates_time.emplace_back(timer.toc() - t0);

    uword new_active = setDiff(active_set, active_set_prev).n_elem;
    ever_active(active_set).fill(true);
    n_active.emplace_back(active_set.n_elem);
    n_new_active.emplace_back(new_active);

    betas.insert_cols(betas.n_cols, beta);

    uword n_ever_active = accu(ever_active);

    if (verbosity >= 1) {
      Rprintf("  ever_active: %i, active: %i, new active: %i\n",
              n_ever_active,
              active_set.n_elem,
              new_active);
    }

    double dev_ratio = 1.0 - dev / null_dev;
    double dev_change = 1.0 - dev / dev_prev;

    if (verbosity >= 1) {
      Rprintf(
        "    dev ratio:  %.3f\n    dev change: %.6f\n", dev_ratio, dev_change);
    }

    // check path stopping criteria
    if (path_length == i ||
        (lambda_type == "auto" &&
         ((i > 1 && dev_change < 1e-5) || dev_ratio >= 0.999 ||
          n_ever_active > n || i == path_length))) {
      hess_time.emplace_back(0);
      it_time.emplace_back(timer.toc() - it_time_i);
      break;
    }

    double lambda_next = lambdas(i);

    if (screening_type == "hessian") {
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
                      X_offset,
                      standardize,
                      verify_hessian,
                      verbosity);

        Hinv_s = Hinv * s(active_perm);
        Hinv_s = Hinv_s(sort_index(active_perm)); // reset permutation
      } else {
        // for logistic regression and no approxiation, simply recompute the
        // hessian and its inverse for the full set of active predictors,
        // since we cannot update the hessian efficiently anyway
        H = model->hessian(X, active_set, X_offset, standardize);

        vec eigval;
        mat eigvec;

        eig_sym(eigval, eigvec, symmatu(H));

        if (eigval.min() < 1e-4 * n) {
          H.diag() += 1e-4 * n;
          eigval += 1e-4 * n;
        }

        Hinv = eigvec * diagmat(1 / eigval) * eigvec.t();
        Hinv_s = Hinv * s(active_set);

        active_perm = active_set;
      }

      hess_time.emplace_back(timer.toc() - t0);

      uvec restricted = abs(c) >= 2 * lambda_next - lambda;

      t0 = timer.toc();

      model->updateGradientOfCorrelation(c_grad,
                                         X,
                                         Hinv_s,
                                         s,
                                         active_set,
                                         find(restricted),
                                         X_offset,
                                         standardize);

      gradcorr_time.emplace_back(timer.toc() - t0);
    }

    if (i > 10 && screening_type == "hessian" && log_hessian_auto) {
      double cd_cum =
        std::accumulate(cd_time.begin() + i - 5, cd_time.end(), 0.0);
      double kkt_cum =
        std::accumulate(kkt_time.begin() + i - 5, kkt_time.end(), 0.0);

      if (verbosity > 0) {
        Rprintf(
          "  CD cum time = %2.4f, KKT cum time = %2.4f\n", cd_cum, kkt_cum);
      }

      if (kkt_cum > 2 * cd_cum) {
        // if Hessian updates take longer than cd updates, switch to approx
        // method
        if (verbosity > 0)
          Rprintf("  NOTE: switching to approx hessian updates\n");

        log_hessian_update_type = "approx";
        model->setLogHessianUpdateType("approx");
        log_hessian_auto = false;

        H = model->hessian(X, active_set, X_offset, standardize);

        vec eigval;
        mat eigvec;
        eig_sym(eigval, eigvec, symmatu(H));

        if (eigval.min() < 1e-4 * n) {
          H.diag() += 1e-4 * n;
          eigval += 1e-4 * n;
        }

        Hinv = eigvec * diagmat(1 / eigval) * eigvec.t();
      }
    }

    strong = abs(c) >= 2 * lambda_next - lambda;
    strong_set = find(strong);
    n_strong.emplace_back(sum(strong));

    screened = screenPredictors(screening_type,
                                strong,
                                ever_active,
                                residual,
                                c,
                                c_grad,
                                X,
                                X_norms_squared,
                                X_offset,
                                y,
                                lambda,
                                lambda_next,
                                gamma,
                                standardize);

    // make sure duplicates stay out
    screened(find(duplicated)).fill(false);

    if (hessian_warm_starts && screening_type == "hessian") {
      beta(active_set) += (lambda - lambda_next) * Hinv_s;
    }

    active_perm_prev = active_perm;
    active_set_prev = active_set;
    lambda_prev = lambda;
    lambda = lambda_next;

    it_time.emplace_back(timer.toc() - it_time_i);

    Rcpp::checkUserInterrupt();
  }

  rescaleCoefficients(betas, X_mean, X_sd, y_center);

  full_time = timer.toc() - full_time;

  umat duplicates_mat = join_horiz(uvec(originals), uvec(duplicates));

  return List::create(Named("beta") = wrap(betas),
                      Named("theta") = wrap(thetas),
                      Named("lambda") = wrap(lambda_out),
                      Named("primals") = wrap(primals),
                      Named("duals") = wrap(duals),
                      Named("dev_ratio") = wrap(dev_ratios),
                      Named("violations") = wrap(n_violations),
                      Named("refits") = wrap(n_refits),
                      Named("active") = wrap(n_active),
                      Named("screened") = wrap(n_screened),
                      Named("new_active") = wrap(n_new_active),
                      Named("passes") = wrap(n_passes),
                      Named("full_time") = wrap(full_time),
                      Named("it_time") = wrap(it_time),
                      Named("cd_time") = wrap(cd_time),
                      Named("hess_time") = wrap(hess_time),
                      Named("kkt_time") = wrap(kkt_time),
                      Named("gradcorr_time") = wrap(gradcorr_time),
                      Named("converged") = wrap(converged),
                      Named("family") = wrap(family));
}

// [[Rcpp::export]]
Rcpp::List
lassoPathDense(arma::mat X,
               arma::vec y,
               const std::string family,
               arma::vec lambdas,
               const std::string lambda_type,
               const bool standardize,
               const std::string screening_type,
               const bool shuffle,
               const arma::uword check_frequency,
               const arma::uword screen_frequency,
               const bool hessian_warm_starts,
               const bool celer_use_old_dual,
               const bool celer_use_accel,
               const bool celer_prune,
               const bool gap_safe_active_start,
               const bool augment_with_gap_safe,
               std::string log_hessian_update_type,
               const arma::uword log_hessian_auto_update_freq,
               const arma::uword path_length,
               const arma::uword maxit,
               const double tol_gap,
               const double gamma,
               const bool store_dual_variables,
               const bool verify_hessian,
               const bool line_search,
               const arma::uword verbosity)
{
  return lassoPath(X,
                   y,
                   family,
                   lambdas,
                   lambda_type,
                   standardize,
                   screening_type,
                   shuffle,
                   check_frequency,
                   screen_frequency,
                   hessian_warm_starts,
                   celer_use_old_dual,
                   celer_use_accel,
                   celer_prune,
                   gap_safe_active_start,
                   augment_with_gap_safe,
                   log_hessian_update_type,
                   log_hessian_auto_update_freq,
                   path_length,
                   maxit,
                   tol_gap,
                   gamma,
                   store_dual_variables,
                   verify_hessian,
                   line_search,
                   verbosity);
}

// [[Rcpp::export]]
Rcpp::List
lassoPathSparse(arma::sp_mat X,
                arma::vec y,
                const std::string family,
                arma::vec lambdas,
                const std::string lambda_type,
                const bool standardize,
                const std::string screening_type,
                const bool shuffle,
                const arma::uword check_frequency,
                const arma::uword screen_frequency,
                const bool hessian_warm_starts,
                const bool celer_use_old_dual,
                const bool celer_use_accel,
                const bool celer_prune,
                const bool gap_safe_active_start,
                const bool augment_with_gap_safe,
                std::string log_hessian_update_type,
                const arma::uword log_hessian_auto_update_freq,
                const arma::uword path_length,
                const arma::uword maxit,
                const double tol_gap,
                const double gamma,
                const bool store_dual_variables,
                const bool verify_hessian,
                const bool line_search,
                const arma::uword verbosity)
{
  return lassoPath(X,
                   y,
                   family,
                   lambdas,
                   lambda_type,
                   standardize,
                   screening_type,
                   shuffle,
                   check_frequency,
                   screen_frequency,
                   hessian_warm_starts,
                   celer_use_old_dual,
                   celer_use_accel,
                   celer_prune,
                   gap_safe_active_start,
                   augment_with_gap_safe,
                   log_hessian_update_type,
                   log_hessian_auto_update_freq,
                   path_length,
                   maxit,
                   tol_gap,
                   gamma,
                   store_dual_variables,
                   verify_hessian,
                   line_search,
                   verbosity);
}

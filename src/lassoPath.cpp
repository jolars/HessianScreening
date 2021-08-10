#include <RcppArmadillo.h>
#include <algorithm>
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
lassoPath(T& X,
          vec& y,
          const std::string family,
          const bool standardize,
          const std::string screening_type,
          const bool hessian_warm_starts,
          const bool gap_safe_active_start,
          std::string log_hessian_update_type,
          const uword log_hessian_auto_update_freq,
          const uword path_length,
          const uword maxit,
          const double tol_infeas,
          double tol_gap,
          const double gamma,
          const bool verify_hessian,
          const bool force_kkt_check,
          const int line_search,
          const uword verbosity)
{
  const uword n = X.n_rows;
  const uword p = X.n_cols;
  std::string stopping_criteria = "celer";

  if (family == "binomial") {
    vec y_unique = sort(unique(y));

    if (y_unique.n_elem != 2) {
      Rcpp::stop("y has more than two unique values");
    } else if (y_unique(0) != 0 || y_unique(1) != 1) {
      Rcpp::stop("y is not in {0, 1}");
    }

    if (screening_type == "edpp")
      Rcpp::stop("EDPP cannot be used in logistic regression");
  }

  bool log_hessian_auto = log_hessian_update_type == "auto";

  if (log_hessian_auto) {
    log_hessian_update_type = "full";
  }

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
      X_norms_squared.fill(static_cast<double>(n));
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
  double lambda_prev = 2 * lambda_max;
  double lambda = lambda_max;

  const double lambda_min_step = 0.1 * min(abs(diff(lambda_grid)));
  double tmp = n < p ? n / path_length : p / path_length;

  uword n_target_nonzero = std::min(p, static_cast<uword>(std::ceil(tmp)));
  n_target_nonzero = std::max(static_cast<uword>(1), n_target_nonzero);

  GetNextLambda getNextLambda{ beta,           c,
                               c_grad,         lambda_grid,
                               lambda_min,     lambda_min_step,
                               screening_type, n_target_nonzero,
                               verbosity };

  std::vector<double> primals;
  std::vector<double> duals;
  std::vector<double> devs;
  std::vector<double> dev_ratios;

  std::vector<uword> n_active;
  std::vector<uword> n_new_active;
  std::vector<uword> n_passes;
  std::vector<uword> n_refits;
  std::vector<double> n_screened;
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

  uvec local_screeen(p, fill::zeros);
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

  bool check_kkt = (screening_type != "gap_safe") || force_kkt_check;

  std::vector<double> it_times;
  std::vector<double> cd_times;
  std::vector<double> kkt_times;
  std::vector<double> gradcorr_times;
  std::vector<double> hess_times;
  std::vector<double> duplicates_times;

  const double null_primal = model->primal(lambda_max, active_set);
  double primal_prev = null_primal;
  // new for dual
  model->updateCorrelation(X, active_set);
  model->updateLinearPredictor(X, active_set);
  model->updateResidual();
  double dual_scale = lambda_max;

  double dual_in = 0;
  wall_clock timer;
  timer.tic();

  double full_time = timer.toc();
  double primal_old = null_primal;
  double tol_gap_global = tol_gap;
  uword i = 0;
  while (true) {
    i++;

    //if( i==3 ){
    //  throw(Rcpp::exception("debug","deubg.cpp",4));
    //}

    double it_time = timer.toc();

    if (verbosity >= 1) {
      Rprintf("step: %i, lambda: %.2f\n", i, lambda);
      Rprintf("******************************\n ITER = %d\n ****************",i);
    }


    vec beta_prev = beta;
    double dev_prev = dev;
    uword n_passes_i_sum = 0;
    uword n_violations_i = 0;
    uword n_refits_i = 0;
    bool first_run = true;

    double cd_time = 0;
    double hess_time = 0;
    double kkt_time = 0;
    if(screening_type == "celer"){
      local_screeen.fill(false);
      uvec beta_non_zero = find(beta != 0);
      primal_prev = model->primal(lambda, beta_non_zero);
      dual_in = model->scaledDual(lambda, dual_scale);
      double dual_gap = primal_prev - dual_in;
      double radius = model->safeScreeningRadius(dual_gap, lambda);

        uvec unscreened_set = find(duplicated == false);

      for (auto&& jj : unscreened_set){
        double d_local =(1 - abs(model->c(jj))/dual_scale)/std::sqrt(X_norms_squared(jj));
        if(d_local > std::sqrt(2)*radius)
          local_screeen(jj) = true;
      }


      tol_gap = 0.3*(primal_prev - dual_in);
      tol_gap= std::max(tol_gap, tol_gap_global);
    }
    while (true) {
      if (verbosity >= 1) {
        Rprintf("  running coordinate descent\n");
      }

      double t0 = timer.toc();

      if (first_run && screening_type != "gap_safe") {
        n_screened.emplace_back(sum(screened));
      }

      auto [primal_value, dual_value, duality_gap, n_passes_i, avg_screened, use_latest_theta, theta_in] =
        model->fit(screened,
                   X,
                   X_norms_squared,
                   lambda,
                   lambda_prev,
                   lambda_max,
                   null_primal,
                   screening_type,
                   gap_safe_active_start,
                   first_run,
                   i,
                   maxit,
                   tol_gap,
                   tol_infeas,
                   line_search,
                   stopping_criteria,
                   verbosity);

      cd_time += timer.toc() - t0;
      n_passes_i_sum += n_passes_i;

      if (screening_type == "gap_safe" &&
          (!first_run || !gap_safe_active_start)) {
        n_screened.push_back(avg_screened);
      }

      t0 = timer.toc();
      int passed_tests = 0;
      if(screening_type == "celer"){
        uvec unscreened_set = find(screened == false && duplicated == false && local_screeen==false);
        model->updateCorrelation(X, unscreened_set);

        violations.fill(false);
        dual_scale = kktCheck(violations, model->c, unscreened_set, lambda);

        dual_scale = std::max(lambda, dual_scale);
        double inner_scale = std::max(lambda, max(abs(model->c(find(screened)))));
        dual_scale = std::max(dual_scale, inner_scale);
        if(verbosity>=1)
          Rcpp::Rcout << "dual_scale/inner_scale=  " << dual_scale/inner_scale << "\n";
        double dual_r = model->scaledDual(lambda, dual_scale);

        vec d_in;
        if(use_latest_theta==0){
          uvec beta_set =  find(beta == 0 && local_screeen==false);
          d_in = model->updateScaleTheta(X, beta_set, theta_in);
          dual_in = model->dual(lambda, theta_in);
          if(verbosity>=1)
            Rcpp::Rcout << "dual_in = " << dual_in << "\n";
          if(dual_in < dual_r)
          {
            dual_in  =dual_r;
            theta_in = model->residual /dual_scale;

            for (auto&& jj : unscreened_set){
              d_in(jj) =(1 - abs(model->c(jj))/dual_scale)/std::sqrt(X_norms_squared(jj));
            }
          }
          double dual_gap = primal_value - dual_in;
          double radius = model->safeScreeningRadius(dual_gap, lambda);
          for (auto&& jj : unscreened_set){
            if(d_in(jj)  > std::sqrt(2)*radius){
              local_screeen(jj) = true;
            }
          }

        }else{
          dual_in  = dual_r;
          d_in.ones(model->p);
          d_in = d_in * (-1);
          uvec beta_set =  find(beta == 0 && local_screeen==false);
          double dual_gap = primal_value - dual_in;
          double radius = model->safeScreeningRadius(dual_gap, lambda);
          for (auto&& jj : beta_set){
            d_in(jj) =(1 - abs(model->c(jj))/dual_scale)/std::sqrt(X_norms_squared(jj));
            if(d_in(jj)  > std::sqrt(2)*radius){
              local_screeen(jj) = true;
            }
          }
        }


        passed_tests = primal_value - dual_in < tol_gap_global * null_primal;

        if(verbosity)
          Rprintf(" Global %d: primal: %.2f, dual: %.2f, duality gap: %.2e\n passed : %d, null_primal: %.2f, tol_gap_global %.2e, lambda %.2f\n",
                  i,
                  primal_value,
                  dual_in,
                  (primal_value - dual_in),
                  passed_tests,
                  null_primal,
                  tol_gap_global,
                  lambda);

        if(passed_tests == 0){
          tol_gap = std::max(0.5*(primal_value - dual_in)/null_primal,tol_gap_global);
          uvec local_screen_set_double = find(duplicated == true  || local_screeen==true);
          if(verbosity>=1){
            Rcpp::Rcout << "active = " << accu(beta!=0) << " of left = " << p -  accu(duplicated == true  || local_screeen==true) << " of total " << p <<"\n";
          }

          for (auto&& jj : local_screen_set_double)
            d_in(jj) =1;
          int working_set_size = 2 * accu( beta != 0);
          if(verbosity>=1)
            Rcpp::Rcout << "working_set_size = " << working_set_size << "\n";
          if( working_set_size == 0)
            working_set_size=1;
          uvec possible_set = duplicated == false && local_screeen==false;
          int index_size= accu(possible_set);

          if( working_set_size >= index_size)
            working_set_size = index_size;

          std::vector<int> index(index_size, 0);
          int counter = 0;
          for (int jj = 0 ; jj != p ; jj++) {
            if(possible_set(jj)){
              index[counter] = jj;
              counter++;
            }
          }
          std::sort(index.begin(),  index.end(),
               [&](const int& a, const int& b) {
                 return (d_in(a) < d_in(b));});

          screened.zeros();
          for(int jj = 0; jj < working_set_size ; jj++){
            if(d_in(index[jj]) < 1)
              screened(index[jj]) = true;
          }


        }else{
          primal_prev = primal_value;
        }


      }else if (check_kkt && !(screening_type == "gap_safe" && first_run)) {
        uvec unscreened_set = find(screened == false && duplicated == false);

        violations.fill(false);

        if (screening_type == "strong" || screening_type == "edpp") {
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
        passed_tests = !any(violations);
      }

      kkt_time += timer.toc() - t0;

      n_violations_i += sum(violations);
      if ( passed_tests && !(screening_type == "gap_safe" &&
                                gap_safe_active_start && first_run)) {
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
                      verbosity);

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

        active_perm = active_set;
      }

      hess_time = timer.toc() - t0;
      hess_times.emplace_back(hess_time);

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

    if (i > 10 && screening_type == "hessian" && log_hessian_auto) {
      double cd_cum =
        std::accumulate(cd_times.begin() + i - 5, cd_times.end(), 0.0);
      double kkt_cum =
        std::accumulate(kkt_times.begin() + i - 5, kkt_times.end(), 0.0);

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

        H = model->hessian(X, active_set);

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

    double lambda_next = getNextLambda(Hinv_s, lambda, active, new_active, i);

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
                                X_mean_scaled,
                                y,
                                lambda,
                                lambda_next,
                                gamma,
                                standardize);

    // make sure duplicates stay out
    screened(find(duplicated)).fill(false);

    if (hessian_warm_starts && hessian_type_screening) {
      beta(active_set) += (lambda - lambda_next) * Hinv_s;
    }

    active_perm_prev = active_perm;
    active_set_prev = active_set;
    lambda_prev = lambda;
    lambda = lambda_next;

    it_times.emplace_back(timer.toc() - it_time);

    Rcpp::checkUserInterrupt();
  }

  active_set = find(beta!=0);
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

// [[Rcpp::export]]
Rcpp::List
lassoPathDense(arma::mat X,
               arma::vec y,
               const std::string family,
               const bool standardize,
               const std::string screening_type,
               const bool hessian_warm_starts,
               const bool gap_safe_active_start,
               std::string log_hessian_update_type,
               const arma::uword log_hessian_auto_update_freq,
               const arma::uword path_length,
               const arma::uword maxit,
               const double tol_infeas,
               const double tol_gap,
               const double gamma,
               const bool verify_hessian,
               const bool force_kkt_check,
               const int line_search,
               const arma::uword verbosity)
{
  return lassoPath(X,
                   y,
                   family,
                   standardize,
                   screening_type,
                   hessian_warm_starts,
                   gap_safe_active_start,
                   log_hessian_update_type,
                   log_hessian_auto_update_freq,
                   path_length,
                   maxit,
                   tol_infeas,
                   tol_gap,
                   gamma,
                   verify_hessian,
                   force_kkt_check,
                   line_search,
                   verbosity);
}

// [[Rcpp::export]]
Rcpp::List
lassoPathSparse(arma::sp_mat X,
                arma::vec y,
                const std::string family,
                const bool standardize,
                const std::string screening_type,
                const bool hessian_warm_starts,
                const bool gap_safe_active_start,
                std::string log_hessian_update_type,
                const arma::uword log_hessian_auto_update_freq,
                const arma::uword path_length,
                const arma::uword maxit,
                const double tol_infeas,
                const double tol_gap,
                const double gamma,
                const bool verify_hessian,
                const bool force_kkt_check,
                const int line_search,
                const arma::uword verbosity)
{
  return lassoPath(X,
                   y,
                   family,
                   standardize,
                   screening_type,
                   hessian_warm_starts,
                   gap_safe_active_start,
                   log_hessian_update_type,
                   log_hessian_auto_update_freq,
                   path_length,
                   maxit,
                   tol_infeas,
                   tol_gap,
                   gamma,
                   verify_hessian,
                   force_kkt_check,
                   line_search,
                   verbosity);
}

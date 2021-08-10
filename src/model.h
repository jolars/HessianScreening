#pragma once

#include <RcppArmadillo.h>

#include "prox.h"

using namespace arma;

class Model
{
public:
  const std::string family;

  vec& y;
  vec& beta;
  vec& residual;
  vec& Xbeta;
  vec& c;

  const vec& X_mean_scaled;
  const vec& X_norms_squared;

  const uword n;
  const uword p;
  const bool standardize;


  Model(const std::string family,
        vec& y,
        vec& beta,
        vec& residual,
        vec& Xbeta,
        vec& c,
        const vec& X_mean_scaled,
        const vec& X_norms_squared,
        const uword n,
        const uword p,
        const bool standardize)
    : family(family)
    , y(y)
    , beta(beta)
    , residual(residual)
    , Xbeta(Xbeta)
    , c(c)
    , X_mean_scaled(X_mean_scaled)
    , X_norms_squared(X_norms_squared)
    , n(n)
    , p(p)
    , standardize(standardize)
  {}

  virtual ~Model() = default;

  void setLogHessianUpdateType(const std::string new_log_hessian_update_type){};

  virtual double primal(const double lambda, const uvec& screened_set) = 0;

  virtual double dual() = 0;
  virtual double dual(const double lambda, const vec & theta) = 0;

  virtual double scaledDual(const double lambda, const double dual_scale) = 0;

  virtual double deviance() = 0;

  virtual mat hessian(const mat& X, const uvec& ind) = 0;

  virtual mat hessian(const sp_mat& X, const uvec& ind) = 0;

  virtual mat hessianUpperRight(const mat& X,
                                const uvec& ind_a,
                                const uvec& ind_b) = 0;

  virtual mat hessianUpperRight(const sp_mat& X,
                                const uvec& ind_a,
                                const uvec& ind_b) = 0;

  virtual double hessianTerm(const mat& X, const uword j) = 0;

  virtual double hessianTerm(const sp_mat& X, const uword j) = 0;

  virtual void updateResidual() = 0;

  virtual void adjustResidual(const mat& X,
                              const uword j,
                              const double beta_diff) = 0;

  virtual void adjustResidual(const sp_mat& X,
                              const uword j,
                              const double beta_diff) = 0;

  void updateLinearPredictor(const mat& X, const uvec& ind)
  {
    Xbeta = X.cols(ind) * beta(ind);
  }

  void updateLinearPredictor(const sp_mat& X, const uvec& ind)
  {
    Xbeta = X.cols(ind) * beta(ind);

    if (standardize)
      Xbeta -= dot(beta(ind), X_mean_scaled(ind));
  }


  vec updateScaleTheta(const mat& X, const uvec& ind, vec & theta)
  {
    double scale = 1.;
    vec d(p,  fill::ones);
    d = d * (-1);
    for (auto&& j : ind) {
      d(j) = abs(dot(X.unsafe_col(j), theta));
      if(d(j) > 1)
        scale = d(j);
    }
    if(scale> 1){}
      theta = theta / scale;

    for (auto&& j : ind) {
      d(j) = (1- d(j)/scale)/std::sqrt(X_norms_squared(j));
    }
    return(d);
  }
  vec updateScaleTheta(const sp_mat& X, const uvec& ind, vec & theta)
  {
    //pair
    double scale = 1.;
    double sum_theta = sum(theta);
    vec d(p,  fill::ones);
    d = d * (-1);
    for (auto&& j : ind) {
      d(j) = dot(X.col(j), theta);
      if(standardize)
        d(j) -= X_mean_scaled(j) * sum_theta;
      d(j) = abs(d(j));
      if(d(j) > 1)
        scale = d(j);
    }
    if(scale> 1)
      theta = theta / scale;
    for (auto&& j : ind) {
      d(j) = 1- d(j)/scale;
    }
    return(d);
  }
  void updateCorrelation(const mat& X, const uvec& ind)
  {
    for (auto&& j : ind) {
      c(j) = dot(X.unsafe_col(j), residual);
    }
  }

  void updateCorrelation(const sp_mat& X, const uvec& ind)
  {
    for (auto&& j : ind) {
      c(j) = dot(X.col(j), residual);
    }

    if (standardize) {
      c(ind) -= X_mean_scaled(ind) * sum(residual);
    }
  }

  inline void updateCorrelation(const mat& X, const uword& j)
  {
    c(j) = dot(X.unsafe_col(j), residual);
  }

  inline void updateCorrelation(const sp_mat& X, const uword& j)
  {
    c(j) = dot(X.col(j), residual);

    if (standardize) {
      c(j) -= X_mean_scaled(j) * accu(residual);
    }
  }

  virtual void updateGradientOfCorrelation(vec& c_grad,
                                           const mat& X,
                                           const vec& Hinv_s,
                                           const vec& s,
                                           const uvec& active_set,
                                           const uvec& restricted_set) = 0;

  virtual void updateGradientOfCorrelation(vec& c_grad,
                                           const sp_mat& X,
                                           const vec& Hinv_s,
                                           const vec& s,
                                           const uvec& active_set,
                                           const uvec& restricted_set) = 0;

  virtual void standardizeY() = 0;

  virtual double safeScreeningRadius(const double duality_gap,
                                     const double lambda) = 0;

  template<typename T>
  void safeScreening(uvec& screened,
                     uvec& screened_set,
                     const T& X,
                     const vec& XTcenter,
                     const double r_screen)
  {
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

  // adopted from https://github.com/tbjohns/BlitzL1/blob/master/src/solver.cpp
  void updatePhi(vec& phi,
                 vec& theta,
                 const uvec& prioritized_features,
                 vec& XTphi,
                 const vec& XTtheta,
                 const double alpha,
                 const double theta_scale)
  {
    // Updates phi via phi = (1-alpha)*phi + alpha*theta*theta_scale
    // Also updates ATphi
    // Requires values of ATtheta and ATphi to be current

    for (auto&& j : prioritized_features) {
      XTphi(j) = (1 - alpha) * XTphi(j) + alpha * theta_scale * XTtheta(j);
    }

    for (uword i = 0; i < phi.n_elem; ++i)
      phi[i] = (1 - alpha) * phi(i) + alpha * theta_scale * theta(i);
  }

  // adopted from https://github.com/tbjohns/BlitzL1/blob/master/src/solver.cpp
  double computeAlpha(const uvec& prioritized_features,
                      const vec& XTphi,
                      const vec& XTtheta,
                      const double lambda,
                      const double theta_scale,
                      const vec& X_norms_squared)
  {
    double best_alpha = 1.0;

    for (auto&& j : prioritized_features) {
      double norm = X_norms_squared(j);

      if (norm <= 0.0)
        continue;

      double l = XTphi(j);
      double r = theta_scale * XTtheta(j);

      if (std::abs(r) <= lambda)
        continue;

      double alpha;

      if (r >= 0)
        alpha = (lambda - l) / (r - l);
      else
        alpha = (-lambda - l) / (r - l);

      if (alpha < best_alpha)
        best_alpha = alpha;
    }

    return best_alpha;
  }

  void prioritizeFeatures(uvec& prioritized_features,
                          vec& feature_priorities,
                          const vec& XTphi,
                          const vec& beta,
                          const vec& X_norms_squared,
                          double lambda)
  {
    // Reorders prioritized_features so that first max_size_C wefk
    // elements are feature indices with highest priority in order

    for (auto&& j : prioritized_features) {
      if (beta(j) != 0) {
        feature_priorities(j) = 0.0;
      } else {
        double norm = std::sqrt(X_norms_squared(j));
        if (norm <= 0) {
          feature_priorities(j) = std::numeric_limits<double>::max();
        } else {
          double priority_value = (lambda - std::abs(XTphi(j))) / norm;
          feature_priorities(j) = priority_value;
        }
      }
    }
    // IndirectComparator cmp(feature_priorities);
    // nth_element(prioritized_features.begin(),
    //             prioritized_features.begin() + max_size_C,
    //             prioritized_features.end(),
    //             cmp);
    // sort(prioritized_features.begin(), prioritized_features.end(), cmp);

    // TODO(jolars): make this more efficient (as above)
    prioritized_features =
      prioritized_features(sort_index(feature_priorities, "ascend"));
  }

  template<typename T>
  std::tuple<double, double, double, uword, double, int, vec> fit(
    uvec& screened,
    const T& X,
    const vec& X_norms_squared,
    const double lambda,
    const double lambda_prev,
    const double lambda_max,
    const double null_primal,
    const std::string screening_type,
    const bool gap_safe_active_start,
    const bool first_run,
    const uword step,
    const uword maxit,
    const double tol_gap,
    const double tol_infeas,
    const int line_search,
    const std::string  stopping_criteria,
    const uword verbosity)
  {
    const uword p = X.n_cols;
    double dual_scale;
    const int f = 10; //celer modulo
    const int K = 5; // celer constant
    int use_latest = 0;
    vec theta(n);

    if (screening_type == "gap_safe" && !first_run && !gap_safe_active_start)
      screened.fill(true);

    uvec screened_set = find(screened);


    // line search parameters
    const double a = 0.1;
    const double b = 0.5;

    vec XTcenter(p);

    vec t(p, fill::ones); // learning rates

    // gap-safe
    const uword screen_interval = 10;

    // blitz
    vec phi(p, fill::zeros);
    vec XTphi(n, fill::zeros);
    vec XTtheta(n, fill::zeros);
    double theta_scale = 1;
    uvec prioritized_features(p);
    vec feature_priorities(p);
    uword working_set_size = 0;

    double n_screened = 0;
    uword it = 0;
    updateLinearPredictor(X, screened_set);
    updateResidual();

    double primal_value = primal(lambda, screened_set);
    double dual_value ;
    updateCorrelation(X, screened_set);
    dual_scale = std::max(lambda, max(abs(c(screened_set))));
    dual_value = scaledDual(lambda, dual_scale);
    theta = residual/dual_scale;

    double duality_gap = primal_value - dual_value;
    if(first_run && duality_gap <= tol_gap * null_primal){
      if (verbosity >= 2){
      Rprintf(" 0:     primal: %.2e, dual: %.2e, duality gap: %.2e\n",
              primal_value,
              dual_value,
              duality_gap / null_primal);
      }
      return { primal_value, dual_value, duality_gap, 0, 0, 1, theta };
    }


    double dual_value_higest = dual_value;

    if (!screened_set.is_empty()) {
      updateLinearPredictor(X, screened_set);
      updateResidual();



      while (it < maxit) {
        it++;

        if (verbosity >= 2) {
          Rprintf("    iter: %i\n", it);
        }

        if (screening_type == "gap_safe" && (it - 1) % screen_interval == 0 &&
            !(first_run && gap_safe_active_start)) {
          if (it > 1) {
            updateLinearPredictor(X, screened_set);
            updateResidual();
          }
          updateCorrelation(X, screened_set);

          dual_scale = std::max(lambda, max(abs(c)));

          primal_value = primal(lambda, screened_set);
          dual_value = scaledDual(lambda, dual_scale);
          duality_gap = primal_value - dual_value;

          double r_screen{ 0 };

          if (screening_type == "gap_safe") {
            XTcenter = c / dual_scale;
            r_screen = safeScreeningRadius(std::max(duality_gap, 0.0), lambda);
          }

          safeScreening(screened, screened_set, X, XTcenter, r_screen);

        } else if (screening_type == "blitz") {
          dual_scale = std::max(lambda, max(abs(c)));

          primal_value = primal(lambda, screened_set);
          dual_value = scaledDual(lambda, dual_scale);
          duality_gap = primal_value - dual_value;

          working_set_size = 2 * accu(beta != 0);

          if (working_set_size < 100)
            working_set_size = 100;

          if (working_set_size > prioritized_features.n_elem)
            working_set_size = prioritized_features.n_elem;

          prioritizeFeatures(prioritized_features,
                             feature_priorities,
                             XTphi,
                             beta,
                             X_norms_squared,
                             lambda);
        }

        n_screened += screened_set.n_elem;

        if (line_search == 3) {
          for (auto&& j : screened_set) {
            updateCorrelation(X, j);
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

                double primal_value_old = primal(lambda, screened_set);

                while (true) {
                  double beta_j_prev = beta(j);

                  beta(j) = beta_j_old + t(j) * v;
                  adjustResidual(X, j, beta(j) - beta_j_prev);

                  primal_value = primal(lambda, screened_set);

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
          uvec screened_set_shuffled = shuffle(screened_set);
          for (auto&& j : screened_set_shuffled) {

            updateCorrelation(X, j);
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
                  primal_value_old = primal(lambda, screened_set);
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
                  updateCorrelation(X, j);
                  if (std::max(c_j_old, c(j)) - std::min(c_j_old, c(j)) >
                      lambda_prev - lambda) {
                    if (verbosity >= 2) {
                      Rprintf(
                        "    linesearch type 2 at iter: %i, index: %i t: %e\n",
                        it,
                        j,
                        t(j));
                    }
                    do_line_search = true;
                    adjustResidual(X, j, beta_j_old - beta(j));
                    beta(j) = beta_j_old;
                    primal_value_old = primal(lambda, screened_set);
                    beta(j) = beta_j_old + t(j) * v;
                    adjustResidual(X, j, beta(j) - beta_j_old);
                  }
                }

                while (do_line_search) {
                  primal_value = primal(lambda, screened_set);

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
        if(stopping_criteria == "celer"){
          primal_value = primal(lambda, screened_set);
          double duality_gap_approx = primal_value - dual_value_higest;
          if( it % f == 0 ){
            updateCorrelation(X, screened_set);
            dual_scale = std::max(lambda, max(abs(c(screened_set))));
            dual_value = scaledDual(lambda, dual_scale);
            if(dual_value > dual_value_higest){
              theta = residual/dual_scale;
              dual_value_higest = dual_value;
              use_latest = 1;
            }else{
              use_latest = 0;
            }

            duality_gap = primal_value - dual_value_higest;

            if (verbosity >= 2)
              Rprintf("      primal: %.2e, dual: %.2e, duality gap: %.2e, duality_gap_approx :%.2e, tol_gap: %.2e\n",
                      primal_value,
                      dual_value_higest,
                      duality_gap / null_primal,
                      duality_gap_approx/null_primal,
                      tol_gap);

            if(duality_gap <= tol_gap * null_primal){
              break;
            }

            Rcpp::checkUserInterrupt();
          }
        }else{
          primal_value = primal(lambda, screened_set);
          dual_value = dual();

          duality_gap = primal_value - dual_value;

          if (verbosity >= 2)
            Rprintf("      primal: %f, dual: %f, duality gap: %f\n",
                    primal_value,
                    dual_value,
                    duality_gap / null_primal);

          if (std::abs(duality_gap) <= tol_gap * null_primal) {
            updateCorrelation(X, screened_set);
            double infeas = lambda > 0 ? max(abs(c(screened_set)) - lambda) : 0;

            if (verbosity >= 2)
              Rprintf("      infeasibility: %f\n", infeas / lambda_max);

            if (infeas <= lambda_max * tol_infeas)
              break;
          }

          Rcpp::checkUserInterrupt();
        }
      }
    } else {
      beta.zeros();
    }

    double avg_screened = n_screened / it;

    return { primal_value, dual_value, duality_gap, it, avg_screened,use_latest, theta };
  }
};

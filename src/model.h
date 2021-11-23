#pragma once

#include "prox.hpp"
#include <RcppArmadillo.h>

class Model
{
public:
  const std::string family;

  arma::vec& y;
  arma::vec& beta;
  arma::vec& residual;
  arma::vec& Xbeta;
  arma::vec& c;

  const arma::vec& X_mean_scaled;
  const arma::vec& X_norms_squared;

  const arma::uword n;
  const arma::uword p;
  const bool standardize;

  Model(const std::string family,
        arma::vec& y,
        arma::vec& beta,
        arma::vec& residual,
        arma::vec& Xbeta,
        arma::vec& c,
        const arma::vec& X_mean_scaled,
        const arma::vec& X_norms_squared,
        const arma::uword n,
        const arma::uword p,
        const bool standardize);

  virtual ~Model() = default;

  void setLogHessianUpdateType(const std::string new_log_hessian_update_type){};

  virtual double primal(const double lambda,
                        const arma::uvec& screened_set) = 0;

  virtual double dual() = 0;
  virtual double dual(const double lambda, const arma::vec& theta) = 0;

  virtual double scaledDual(const double lambda, const double dual_scale) = 0;

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

  void updateLinearPredictor(const arma::mat& X, const arma::uvec& ind);

  void updateLinearPredictor(const arma::sp_mat& X, const arma::uvec& ind);

  arma::vec updateScaleTheta(const arma::mat& X,
                             const arma::uvec& ind,
                             arma::vec& theta);

  arma::vec updateScaleTheta(const arma::sp_mat& X,
                             const arma::uvec& ind,
                             arma::vec& theta);

  void updateCorrelation(const arma::mat& X, const arma::uvec& ind);

  void updateCorrelation(const arma::sp_mat& X, const arma::uvec& ind);

  void updateCorrelation(const arma::mat& X, const arma::uword& j);

  void updateCorrelation(const arma::sp_mat& X, const arma::uword& j);

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

  template<typename T>
  void safeScreening(arma::uvec& screened,
                     arma::uvec& screened_set,
                     const T& X,
                     const arma::vec& XTcenter,
                     const double r_screen)
  {
    for (auto&& j : screened_set) {
      double r_normX_j = r_screen * std::sqrt(X_norms_squared(j));

      if (r_normX_j >= 1) {
        continue;
      }

      if (std::abs(XTcenter(j)) + r_normX_j + std::sqrt(arma::datum::eps) < 1) {
        // predictor must be zero; update residual and remove from screened set
        if (beta(j) != 0) {
          adjustResidual(X, j, -beta(j));
          beta(j) = 0;
        }

        screened(j) = false;
      }
    }

    screened_set = arma::find(screened);
  }

  void updatePhi(arma::vec& phi,
                 arma::vec& theta,
                 const arma::uvec& prioritized_features,
                 arma::vec& XTphi,
                 const arma::vec& XTtheta,
                 const double alpha,
                 const double theta_scale);

  double computeAlpha(const arma::uvec& prioritized_features,
                      const arma::vec& XTphi,
                      const arma::vec& XTtheta,
                      const double lambda,
                      const double theta_scale,
                      const arma::vec& X_norms_squared);

  void prioritizeFeatures(arma::uvec& prioritized_features,
                          arma::vec& feature_priorities,
                          const arma::vec& XTphi,
                          const arma::vec& beta,
                          const arma::vec& X_norms_squared,
                          double lambda);

  template<typename T>
  std::tuple<double, double, double, arma::uword, double, int, arma::vec> fit(
    arma::uvec& screened,
    const T& X,
    const arma::vec& X_norms_squared,
    const double lambda,
    const double lambda_prev,
    const double lambda_max,
    const double null_primal,
    const std::string screening_type,
    const bool gap_safe_active_start,
    const bool first_run,
    const arma::uword step,
    const arma::uword maxit,
    const double tol_gap,
    const double tol_infeas,
    const int line_search,
    const std::string stopping_criteria,
    const arma::uword verbosity)
  {
    using namespace arma;

    const uword p = X.n_cols;
    double dual_scale;
    const int f = 10; // celer modulo
    const int K = 5;  // celer constant
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
    // double theta_scale = 1;
    uvec prioritized_features(p);
    vec feature_priorities(p);
    uword working_set_size = 0;

    double n_screened = 0;
    uword it = 0;
    updateLinearPredictor(X, screened_set);
    updateResidual();

    double primal_value = primal(lambda, screened_set);
    double dual_value;
    updateCorrelation(X, screened_set);
    dual_scale = std::max(lambda, max(abs(c(screened_set))));
    dual_value = scaledDual(lambda, dual_scale);
    theta = residual / dual_scale;

    double duality_gap = primal_value - dual_value;
    if (first_run && duality_gap <= tol_gap * null_primal) {
      if (verbosity >= 2) {
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
        if (stopping_criteria == "celer") {
          primal_value = primal(lambda, screened_set);
          double duality_gap_approx = primal_value - dual_value_higest;
          if (it % f == 0) {
            updateCorrelation(X, screened_set);
            dual_scale = std::max(lambda, max(abs(c(screened_set))));
            dual_value = scaledDual(lambda, dual_scale);
            if (dual_value > dual_value_higest) {
              theta = residual / dual_scale;
              dual_value_higest = dual_value;
              use_latest = 1;
            } else {
              use_latest = 0;
            }

            duality_gap = primal_value - dual_value_higest;

            if (verbosity >= 2)
              Rprintf("      primal: %.2e, dual: %.2e, duality gap: %.2e, "
                      "duality_gap_approx :%.2e, tol_gap: %.2e\n",
                      primal_value,
                      dual_value_higest,
                      duality_gap / null_primal,
                      duality_gap_approx / null_primal,
                      tol_gap);

            if (duality_gap <= tol_gap * null_primal) {
              break;
            }

            Rcpp::checkUserInterrupt();
          }
        } else {
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

    return { primal_value, dual_value, duality_gap, it,
             avg_screened, use_latest, theta };
  }
};

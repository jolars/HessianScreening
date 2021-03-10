#pragma once

#include <RcppArmadillo.h>

#include "prox.h"

using namespace arma;

class Model
{
public:
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

  double dual_scale{ 0 };

  Model(vec& y,
        vec& beta,
        vec& residual,
        vec& Xbeta,
        vec& c,
        const vec& X_mean_scaled,
        const vec& X_norms_squared,
        const uword n,
        const uword p,
        const bool standardize)
    : y(y)
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

  virtual double primal(const double lambda, const uvec& screened_set) = 0;

  virtual double dual() = 0;

  virtual double scaledDual(const double lambda) = 0;

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

  void updateCorrelation(const mat& X, const uvec& ind)
  {
#pragma omp parallel for
    for (auto&& j : ind) {
      c(j) = dot(X.unsafe_col(j), residual);
    }
  }

  void updateCorrelation(const sp_mat& X, const uvec& ind)
  {
#pragma omp parallel for
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
  uvec safeScreening(uvec& screened,
                     const T& X,
                     const vec& XTcenter,
                     const double r_screen)
  {
    for (auto&& j : screened) {
      double r_normX_j = r_screen * std::sqrt(X_norms_squared(j));

      if (r_normX_j >= 1) {
        continue;
      }

      if (std::abs(XTcenter(j)) + r_normX_j < 1) {
        // predictor must be zero; update residual and remove from screened set
        if (beta(j) != 0) {
          adjustResidual(X, j, -beta(j));
          beta(j) = 0;
        }

        screened(j) = false;
      }
    }

    return find(screened);
  }

  template<typename T>
  std::tuple<double, double, double, uword, double> fit(
    uvec& screened,
    const T& X,
    const vec& X_norms_squared,
    const double lambda,
    const double lambda_max,
    const double null_deviance,
    const std::string screening_type,
    const uword maxit,
    const double tol_decr,
    const double tol_gap,
    const double tol_infeas,
    const uword verbosity)
  {
    const uword p = X.n_cols;

    if (screening_type == "gap_safe") {
      screened.fill(true);
    }

    uvec screened_set = find(screened);

    double primal_value = primal(lambda, screened_set);
    double dual_value = dual();
    double duality_gap = primal_value - dual_value;

    vec XTcenter(p);

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
          updateCorrelation(X, screened_set);

          dual_scale = std::max(lambda, max(abs(c)));

          primal_value = primal(lambda, screened_set);
          dual_value = scaledDual(lambda);
          duality_gap = primal_value - dual_value + datum::eps;

          double r_screen{ 0 };

          if (screening_type == "gap_safe") {
            XTcenter = c / dual_scale;
            r_screen = safeScreeningRadius(duality_gap, lambda);
          }

          screened_set = safeScreening(screened, X, XTcenter, r_screen);
          n_screened += screened_set.n_elem;
        }

        vec beta_screened_old = beta(screened_set);
        double primal_value_old = primal(lambda, screened_set);
        double dual_value_old = dual();

        double primal_value_old2 = primal(lambda, screened_set);
        for (auto&& j : screened_set) {
          double beta_j_old = beta(j);
          updateCorrelation(X, j);
          double hess_j = hessianTerm(X, j);
          beta(j) = prox(c(j) / hess_j + beta_j_old, lambda / hess_j);

          if (beta_j_old != beta(j)) {
            adjustResidual(X, j, beta(j) - beta_j_old);
          }
        }

        primal_value = primal(lambda, screened_set);
        dual_value = dual();

        double primal_value_change = primal_value - primal_value_old;

        double t = 1;
        vec beta_screened = beta(screened_set);

        uword line_it = 0;

        while (primal_value >= primal_value_old &&
               dual_value <= dual_value_old && line_it < 15) {
          line_it++;
          t *= 0.5;

          beta(screened_set) = (1 - t) * beta_screened_old + t * beta_screened;

          updateLinearPredictor(X, screened_set);
          updateResidual();

          primal_value = primal(lambda, screened_set);
          dual_value = dual();
        }

        primal_value_change = primal_value - primal_value_old;

        if (verbosity >= 2) {
          Rprintf("      primal: %f, dual: %f, primal_change: %f\n",
                  primal_value,
                  dual_value,
                  primal_value_change);
          if(primal_value_change >20)
            Rcpp::stop("inverse matrix computation is incorrect");
        }

        if (abs(primal_value_change) <= tol_gap * primal_value) {
          dual_value = dual();
          duality_gap = std::abs(primal_value - dual_value);

          updateCorrelation(X, screened_set);

          double max_infeas =
            lambda > 0 ? max(abs(c(screened_set)) - lambda) : 0;

          if (verbosity >= 2) {
            Rprintf("      infeasibility: %f, duality gap: %f\n",
                    max_infeas,
                    duality_gap);
          }

          if (max_infeas <= lambda_max * tol_infeas &&
              duality_gap <= tol_gap * null_deviance) {
            break;
          }
        }

        Rcpp::checkUserInterrupt();

        it++;
      }
    } else {
      beta.zeros();
    }

    double avg_screened = n_screened / static_cast<double>(it);

    return { primal_value, dual_value, duality_gap, it, avg_screened };
  }
};

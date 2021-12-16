#include "binomial.h"
#include "prox.h"
#include "utils.h"

Binomial::Binomial(const std::string family,
                   arma::vec& y,
                   arma::vec& beta,
                   arma::vec& residual,
                   arma::vec& Xbeta,
                   const arma::vec& X_mean_scaled,
                   const arma::vec& X_norms_squared,
                   const arma::uword n,
                   const arma::uword p,
                   const bool standardize,
                   const std::string log_hessian_update_type)
  : Model{ family,          y, beta, residual,   Xbeta, X_mean_scaled,
           X_norms_squared, n, p,    standardize }
  , expXbeta(y.n_elem, arma::fill::zeros)
  , pr(y.n_elem, arma::fill::zeros)
  , w(y.n_elem, arma::fill::zeros)
  , log_hessian_update_type{ log_hessian_update_type }
{}

void
Binomial::setLogHessianUpdateType(const std::string new_log_hessian_update_type)
{
  log_hessian_update_type = new_log_hessian_update_type;
};

double
Binomial::primal(const double lambda)
{
  using namespace arma;

  return -sum(y % Xbeta - log1p(expXbeta)) + lambda * norm(beta, 1);
}

double
Binomial::primal(const double lambda, const arma::uvec& screened_set)
{
  using namespace arma;

  return -sum(y % Xbeta - log1p(expXbeta)) +
         lambda * norm(beta(screened_set), 1);
}

double
Binomial::dual()
{
  return -arma::sum(pr % arma::log(pr) + (1.0 - pr) % arma::log(1.0 - pr));
}

double
Binomial::scaledDual(const double lambda)
{
  using namespace arma;

  if (dual_scale == 0) {
    return 0;
  } else {
    double alpha = lambda / dual_scale;

    vec prx = clamp(y - alpha * residual, p_min, p_max);

    return -sum(prx % log(prx) + (1 - prx) % log(1 - prx));
  }
}

double
Binomial::deviance()
{
  return -2 * arma::sum(y % Xbeta - arma::log1p(expXbeta));
}

double
Binomial::hessianTerm(const arma::mat& X, const arma::uword j)
{
  return std::max(arma::dot(arma::square(X.col(j)), w),
                  std::sqrt(arma::datum::eps));
}

double
Binomial::hessianTerm(const arma::sp_mat& X, const arma::uword j)
{
  double out = arma::dot(arma::square(X.col(j)), w);

  if (standardize) {
    out += std::pow(X_mean_scaled(j), 2) * arma::accu(w) -
           2 * arma::dot(X.col(j), w) * X_mean_scaled(j);
  }

  return std::max(out, std::sqrt(arma::datum::eps));
}

void
Binomial::updateResidual()
{
  expXbeta = arma::exp(Xbeta);
  pr = arma::clamp(expXbeta / (1 + expXbeta), p_min, p_max);
  w = pr % (1 - pr);
  residual = y - pr;
}

void
Binomial::adjustResidual(const arma::mat& X,
                         const arma::uword j,
                         const double beta_diff)
{
  Xbeta += X.col(j) * beta_diff;
  updateResidual();
}

void
Binomial::adjustResidual(const arma::sp_mat& X,
                         const arma::uword j,
                         const double beta_diff)
{
  Xbeta += X.col(j) * beta_diff;

  if (standardize)
    Xbeta -= X_mean_scaled(j) * beta_diff;

  updateResidual();
}

arma::mat
Binomial::hessian(const arma::mat& X, const arma::uvec& ind)
{
  if (log_hessian_update_type == "approx") {
    return 0.25 * X.cols(ind).t() * X.cols(ind);
  } else {
    return X.cols(ind).t() * arma::diagmat(w) * X.cols(ind);
  }
}

arma::mat
Binomial::hessian(const arma::sp_mat& X, const arma::uvec& ind)
{
  using namespace arma;

  if (log_hessian_update_type == "approx") {
    mat H = conv_to<mat>::from(X.cols(ind).t() * X.cols(ind));

    if (standardize)
      H -= X.n_rows * X_mean_scaled(ind) * X_mean_scaled(ind).t();

    H *= 0.25;

    return H;
  } else {
    mat D = diagmat(w);
    mat H = conv_to<mat>::from(X.cols(ind).t() * D * X.cols(ind));

    if (standardize) {
      mat XmDX = X_mean_scaled(ind) * sum(D * X.cols(ind), 0);
      H +=
        sum(w) * X_mean_scaled(ind) * X_mean_scaled(ind).t() - XmDX - XmDX.t();
    }

    return H;
  }
}

arma::mat
Binomial::hessianUpperRight(const arma::mat& X,
                            const arma::uvec& ind_a,
                            const arma::uvec& ind_b)
{
  if (log_hessian_update_type == "approx") {
    return 0.25 * X.cols(ind_a).t() * X.cols(ind_b);
  } else {
    return X.cols(ind_a).t() * arma::diagmat(w) * X.cols(ind_b);
  }
}

arma::mat
Binomial::hessianUpperRight(const arma::sp_mat& X,
                            const arma::uvec& ind_a,
                            const arma::uvec& ind_b)
{
  using namespace arma;

  mat H(ind_a.n_elem, ind_b.n_elem);

  if (log_hessian_update_type == "approx") {
    if (ind_b.n_elem == 1) {
      uword i = 0;
      for (auto&& j : ind_a) {
        H(i, 0) = dot(X.col(j), X.col(as_scalar(ind_b)));
        i++;
      }
    } else {
      H = X.cols(ind_a).t() * X.cols(ind_b);
    }

    if (standardize)
      H -= X.n_rows * X_mean_scaled(ind_a) * X_mean_scaled(ind_b).t();

    return 0.25 * H;

  } else {
    mat D = diagmat(w);

    if (ind_b.n_elem == 1) {
      uword i = 0;
      sp_mat w_Xb = w % X.col(as_scalar(ind_b));
      for (auto&& j : ind_a) {
        H(i, 0) = dot(X.col(j), w_Xb);
        i++;
      }
    } else {
      H = X.cols(ind_a).t() * D * X.cols(ind_b);
    }

    if (standardize) {
      mat XamDXb = X_mean_scaled(ind_a) * sum(D * X.cols(ind_b));
      mat XbmDXa = X_mean_scaled(ind_b) * sum(D * X.cols(ind_a));
      H += sum(w) * X_mean_scaled(ind_a) * X_mean_scaled(ind_b).t() -
           XbmDXa.t() - XamDXb;
    }
  }

  return H;
}

void
Binomial::updateGradientOfCorrelation(arma::vec& c_grad,
                                      const arma::mat& X,
                                      const arma::vec& Hinv_s,
                                      const arma::vec& s,
                                      const arma::uvec& active_set,
                                      const arma::uvec& restricted_set)
{
  using namespace arma;

  uvec inactive_restricted = setDiff(restricted_set, active_set);

  const vec tmp = w % (X.cols(active_set) * Hinv_s);

  c_grad.zeros();

  for (auto&& j : inactive_restricted) {
    c_grad(j) = dot(X.unsafe_col(j), tmp);
  }

  c_grad(active_set) = s(active_set);
}

void
Binomial::updateGradientOfCorrelation(arma::vec& c_grad,
                                      const arma::sp_mat& X,
                                      const arma::vec& Hinv_s,
                                      const arma::vec& s,
                                      const arma::uvec& active_set,
                                      const arma::uvec& restricted_set)
{
  using namespace arma;

  uvec inactive_restricted = setDiff(restricted_set, active_set);

  const vec dsq = sqrt(w);

  c_grad.zeros();

  if (standardize) {
    const mat Dsq = diagmat(dsq);
    const mat Dsq_X = Dsq * X.cols(inactive_restricted);
    const mat Dsq_Xa = Dsq * X.cols(active_set);

    const mat dsq_mu = dsq * X_mean_scaled(inactive_restricted).t();
    const mat dsq_mua_Hinv_s = dsq * (X_mean_scaled(active_set).t() * Hinv_s);
    const mat Dsq_Xa_Hinv_s = Dsq_Xa * Hinv_s;

    c_grad(inactive_restricted) = Dsq_X.t() * (Dsq_Xa_Hinv_s - dsq_mua_Hinv_s) +
                                  dsq_mu.t() * (dsq_mua_Hinv_s - Dsq_Xa_Hinv_s);

  } else {
    const vec tmp = w % (X.cols(active_set) * Hinv_s);

    for (auto&& j : inactive_restricted) {
      c_grad(j) = dot(X.col(j), tmp);
    }
  }

  c_grad(active_set) = s(active_set);
}

void
Binomial::standardizeY()
{}

double
Binomial::safeScreeningRadius(const double duality_gap, const double lambda)
{
  return std::sqrt(2 * duality_gap) / (2 * lambda);
}

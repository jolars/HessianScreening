#include "binomial.h"
#include "prox.h"
#include "utils.h"

Binomial::Binomial(const std::string family,
                   const arma::uword n,
                   const std::string log_hessian_update_type)
  : Model{ family }
  , expXbeta(n, arma::fill::zeros)
  , pr(n, arma::fill::zeros)
  , w(n, arma::fill::zeros)
  , log_hessian_update_type{ log_hessian_update_type }
{}

void
Binomial::setLogHessianUpdateType(const std::string new_log_hessian_update_type)
{
  log_hessian_update_type = new_log_hessian_update_type;
}

double
Binomial::primal(const arma::vec& residual,
                 const arma::vec& Xbeta,
                 const arma::vec& beta,
                 const arma::vec& y,
                 const double lambda)
{
  using namespace arma;

  return -sum(y % Xbeta - log1p(expXbeta)) + lambda * norm(beta, 1);
}

double
Binomial::primal(const arma::vec& residual,
                 const arma::vec& Xbeta,
                 const arma::vec& beta,
                 const arma::vec& y,
                 const double lambda,
                 const arma::uvec& screened_set)
{
  using namespace arma;

  return -sum(y % Xbeta - log1p(expXbeta)) +
         lambda * norm(beta(screened_set), 1);
}

double
Binomial::dual(const arma::vec& theta, const arma::vec& y, const double lambda)
{
  using namespace arma;

  vec prx = clamp(y - lambda * theta, p_min, p_max);

  return -sum(prx % log(prx) + (1 - prx) % log(1 - prx));
}

double
Binomial::deviance(const arma::vec& residual,
                   const arma::vec& Xbeta,
                   const arma::vec& y)
{
  return -2 * arma::sum(y % Xbeta - arma::log1p(expXbeta));
}

double
Binomial::hessianTerm(const arma::mat& X,
                      const arma::uword j,
                      const arma::vec& X_offset,
                      const bool standardize)
{
  return arma::dot(arma::square(X.col(j)), w);
}

double
Binomial::hessianTerm(const arma::sp_mat& X,
                      const arma::uword j,
                      const arma::vec& X_offset,
                      const bool standardize)
{
  double out = arma::dot(arma::square(X.col(j)), w);

  if (standardize) {
    out += std::pow(X_offset(j), 2) * arma::accu(w) -
           2 * arma::dot(X.col(j), w) * X_offset(j);
  }

  return out;
}

void
Binomial::updateResidual(arma::vec& residual,
                         const arma::vec& Xbeta,
                         const arma::vec& y)
{
  expXbeta = arma::exp(Xbeta);
  pr = arma::clamp(expXbeta / (1 + expXbeta), p_min, p_max);
  w = pr % (1 - pr);
  residual = y - pr;
}

void
Binomial::adjustResidual(arma::vec& residual,
                         arma::vec& Xbeta,
                         const arma::mat& X,
                         const arma::vec& y,
                         const arma::uword j,
                         const double beta_diff,
                         const arma::vec& X_offset,
                         const bool standardize)
{
  Xbeta += X.col(j) * beta_diff;
  updateResidual(residual, Xbeta, y);
}

void
Binomial::adjustResidual(arma::vec& residual,
                         arma::vec& Xbeta,
                         const arma::sp_mat& X,
                         const arma::vec& y,
                         const arma::uword j,
                         const double beta_diff,
                         const arma::vec& X_offset,
                         const bool standardize)
{
  Xbeta += X.col(j) * beta_diff;

  if (standardize)
    Xbeta -= X_offset(j) * beta_diff;

  updateResidual(residual, Xbeta, y);
}

arma::vec
Binomial::weights(const arma::vec& residual, const arma::vec& y)
{
  const arma::vec pr = y - residual;

  return pr % (1 - pr);
}

arma::mat
Binomial::hessian(const arma::mat& X,
                  const arma::uvec& ind,
                  const arma::vec& X_offset,
                  const bool standardize)
{
  if (log_hessian_update_type == "approx") {
    return 0.25 * X.cols(ind).t() * X.cols(ind);
  } else {
    return X.cols(ind).t() * arma::diagmat(w) * X.cols(ind);
  }
}

arma::mat
Binomial::hessian(const arma::sp_mat& X,
                  const arma::uvec& ind,
                  const arma::vec& X_offset,
                  const bool standardize)
{
  using namespace arma;

  if (log_hessian_update_type == "approx") {
    mat H = conv_to<mat>::from(X.cols(ind).t() * X.cols(ind));

    if (standardize)
      H -= X.n_rows * X_offset(ind) * X_offset(ind).t();

    H *= 0.25;

    return H;
  } else {
    mat D = diagmat(w);
    mat H = conv_to<mat>::from(X.cols(ind).t() * D * X.cols(ind));

    if (standardize) {
      mat XmDX = X_offset(ind) * sum(D * X.cols(ind), 0);
      H += accu(w) * X_offset(ind) * X_offset(ind).t() - XmDX - XmDX.t();
    }

    return H;
  }
}

arma::mat
Binomial::hessianUpperRight(const arma::mat& X,
                            const arma::uvec& ind_a,
                            const arma::uvec& ind_b,
                            const arma::vec& X_offset,
                            const bool standardize)
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
                            const arma::uvec& ind_b,
                            const arma::vec& X_offset,
                            const bool standardize)
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
      H -= X.n_rows * X_offset(ind_a) * X_offset(ind_b).t();

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
      mat XamDXb = X_offset(ind_a) * sum(D * X.cols(ind_b));
      mat XbmDXa = X_offset(ind_b) * sum(D * X.cols(ind_a));
      H += sum(w) * X_offset(ind_a) * X_offset(ind_b).t() - XbmDXa.t() - XamDXb;
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
                                      const arma::uvec& restricted_set,
                                      const arma::vec& X_offset,
                                      const bool standardize)
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
                                      const arma::uvec& restricted_set,
                                      const arma::vec& X_offset,
                                      const bool standardize)
{
  using namespace arma;

  uvec inactive_restricted = setDiff(restricted_set, active_set);

  const vec dsq = sqrt(w);

  c_grad.zeros();

  if (standardize) {
    const mat Dsq = diagmat(dsq);
    const mat Dsq_X = Dsq * X.cols(inactive_restricted);
    const mat Dsq_Xa = Dsq * X.cols(active_set);

    const mat dsq_mu = dsq * X_offset(inactive_restricted).t();
    const mat dsq_mua_Hinv_s = dsq * (X_offset(active_set).t() * Hinv_s);
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
Binomial::standardizeY(arma::vec& y)
{}

double
Binomial::safeScreeningRadius(const double duality_gap, const double lambda)
{
  return std::sqrt(0.5 * std::max(duality_gap, 0.0)) / lambda;
}

double
Binomial::toleranceModifier(const arma::vec& y)
{
  return y.n_elem * std::log(2);
}

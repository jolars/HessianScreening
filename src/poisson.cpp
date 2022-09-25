#include "poisson.h"
#include "prox.h"
#include "utils.h"

Poisson::Poisson(const std::string family, const arma::uword n)
  : Model{ family }
  , expXbeta(n, arma::fill::zeros)
  , w(n, arma::fill::ones)
{
}

double
Poisson::primal(const arma::vec& residual,
                const arma::vec& Xbeta,
                const arma::vec& beta,
                const arma::vec& y,
                const double lambda)
{
  using namespace arma;

  return accu(expXbeta - y % Xbeta + lgamma(y + 1)) +
         lambda * norm(beta, 1);
}

double
Poisson::primal(const arma::vec& residual,
                const arma::vec& Xbeta,
                const arma::vec& beta,
                const arma::vec& y,
                const double lambda,
                const arma::uvec& screened_set)
{
  using namespace arma;

  return accu(expXbeta - y % Xbeta + lgamma(y + 1)) +
         lambda * norm(beta(screened_set), 1);
}

double
Poisson::dual(const arma::vec& theta,
              const arma::vec& y,
              const double dual_scale,
              const double lambda)
{
  using namespace arma;

  double s = lambda / dual_scale;

  return accu((y * s - theta * lambda) %
                (1 - trunc_log(y * s - theta * lambda)) +
              lgamma(y * s + 1));
}

double
Poisson::deviance(const arma::vec& residual,
                  const arma::vec& Xbeta,
                  const arma::vec& y)
{
  return -2 * arma::accu(expXbeta - y % Xbeta + lgamma(y + 1));
}

double
Poisson::hessianTerm(const arma::mat& X,
                     const arma::uword j,
                     const arma::vec& X_offset,
                     const bool standardize)
{
  return arma::dot(arma::square(X.col(j)), w);
}

double
Poisson::hessianTerm(const arma::sp_mat& X,
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
Poisson::updateResidual(arma::vec& residual,
                        const arma::vec& Xbeta,
                        const arma::vec& y)
{
  expXbeta = arma::trunc_exp(Xbeta);
  w = expXbeta;
  residual = y - expXbeta;
}

void
Poisson::adjustResidual(arma::vec& residual,
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
Poisson::adjustResidual(arma::vec& residual,
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
Poisson::weights(const arma::vec& residual, const arma::vec& y)
{
  return y - residual;
}

arma::mat
Poisson::hessian(const arma::mat& X,
                 const arma::uvec& ind,
                 const arma::vec& X_offset,
                 const bool standardize)
{
  return X.cols(ind).t() * arma::diagmat(w) * X.cols(ind);
}

arma::mat
Poisson::hessian(const arma::sp_mat& X,
                 const arma::uvec& ind,
                 const arma::vec& X_offset,
                 const bool standardize)
{
  using namespace arma;

  mat H = conv_to<mat>::from(X.cols(ind).t() * diagmat(w) * X.cols(ind));

  if (standardize) {
    mat XmDX = X_offset(ind) * sum(diagmat(w) * X.cols(ind), 0);
    H += accu(w) * X_offset(ind) * X_offset(ind).t() - XmDX - XmDX.t();
  }

  return H;
}

arma::mat
Poisson::hessianUpperRight(const arma::mat& X,
                           const arma::uvec& ind_a,
                           const arma::uvec& ind_b,
                           const arma::vec& X_offset,
                           const bool standardize)
{
  return X.cols(ind_a).t() * arma::diagmat(w) * X.cols(ind_b);
}

arma::mat
Poisson::hessianUpperRight(const arma::sp_mat& X,
                           const arma::uvec& ind_a,
                           const arma::uvec& ind_b,
                           const arma::vec& X_offset,
                           const bool standardize)
{
  using namespace arma;

  mat H(ind_a.n_elem, ind_b.n_elem);

  if (ind_b.n_elem == 1) {
    uword i = 0;
    sp_mat w_Xb = w % X.col(as_scalar(ind_b));
    for (auto&& j : ind_a) {
      H(i, 0) = dot(X.col(j), w_Xb);
      i++;
    }
  } else {
    H = X.cols(ind_a).t() * diagmat(w) * X.cols(ind_b);
  }

  if (standardize) {
    mat XamDXb = X_offset(ind_a) * sum(diagmat(w) * X.cols(ind_b));
    mat XbmDXa = X_offset(ind_b) * sum(diagmat(w) * X.cols(ind_a));
    H += sum(w) * X_offset(ind_a) * X_offset(ind_b).t() - XbmDXa.t() - XamDXb;
  }

  return H;
}

void
Poisson::updateGradientOfCorrelation(arma::vec& c_grad,
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
Poisson::updateGradientOfCorrelation(arma::vec& c_grad,
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
    const mat Dsq_X = diagmat(dsq) * X.cols(inactive_restricted);
    const mat Dsq_Xa = diagmat(dsq) * X.cols(active_set);

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
Poisson::standardizeY(arma::vec& y)
{
}

double
Poisson::safeScreeningRadius(const double duality_gap, const double lambda)
{
  Rcpp::stop("Gap-Safe screening does not work for Poisson loss");
}

double
Poisson::toleranceModifier(const arma::vec& y)
{
  return static_cast<double>(y.n_elem) + arma::accu(arma::lgamma(y + 1));
}

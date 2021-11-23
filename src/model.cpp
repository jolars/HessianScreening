#include "model.h"
#include "prox.h"
#include <RcppArmadillo.h>

Model::Model(const std::string family,
             arma::vec& y,
             arma::vec& beta,
             arma::vec& residual,
             arma::vec& Xbeta,
             arma::vec& c,
             const arma::vec& X_mean_scaled,
             const arma::vec& X_norms_squared,
             const arma::uword n,
             const arma::uword p,
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

void
Model::updateLinearPredictor(const arma::mat& X, const arma::uvec& ind)
{
  Xbeta = X.cols(ind) * beta(ind);
}

void
Model::updateLinearPredictor(const sp_mat& X, const uvec& ind)
{
  Xbeta = X.cols(ind) * beta(ind);

  if (standardize)
    Xbeta -= arma::dot(beta(ind), X_mean_scaled(ind));
}

arma::vec
Model::updateScaleTheta(const arma::mat& X,
                        const arma::uvec& ind,
                        arma::vec& theta)
{
  using namespace arma;

  double scale = 1.;
  vec d(p, fill::ones);
  d = d * (-1);
  for (auto&& j : ind) {
    d(j) = abs(dot(X.unsafe_col(j), theta));
    if (d(j) > 1)
      scale = d(j);
  }
  if (scale > 1) {
  }
  theta = theta / scale;

  for (auto&& j : ind) {
    d(j) = (1 - d(j) / scale) / std::sqrt(X_norms_squared(j));
  }
  return (d);
}

arma::vec
Model::updateScaleTheta(const arma::sp_mat& X,
                        const arma::uvec& ind,
                        arma::vec& theta)
{
  using namespace arma;

  // pair
  double scale = 1.;
  double sum_theta = sum(theta);
  vec d(p, fill::ones);
  d = d * (-1);

  for (auto&& j : ind) {
    d(j) = dot(X.col(j), theta);
    if (standardize)
      d(j) -= X_mean_scaled(j) * sum_theta;
    d(j) = abs(d(j));
    if (d(j) > 1)
      scale = d(j);
  }
  if (scale > 1)
    theta = theta / scale;
  for (auto&& j : ind) {
    d(j) = 1 - d(j) / scale;
  }
  return (d);
}

void
Model::updateCorrelation(const arma::mat& X, const arma::uvec& ind)
{
  for (auto&& j : ind) {
    c(j) = arma::dot(X.unsafe_col(j), residual);
  }
}

void
Model::updateCorrelation(const arma::sp_mat& X, const arma::uvec& ind)
{
  for (auto&& j : ind) {
    c(j) = arma::dot(X.col(j), residual);
  }

  if (standardize) {
    c(ind) -= X_mean_scaled(ind) * arma::sum(residual);
  }
}

void
Model::updateCorrelation(const arma::mat& X, const arma::uword& j)
{
  c(j) = arma::dot(X.unsafe_col(j), residual);
}

void
Model::updateCorrelation(const arma::sp_mat& X, const arma::uword& j)
{
  c(j) = arma::dot(X.col(j), residual);

  if (standardize) {
    c(j) -= X_mean_scaled(j) * arma::accu(residual);
  }
}

// adopted from https://github.com/tbjohns/BlitzL1/blob/master/src/solver.cpp
void
Model::updatePhi(arma::vec& phi,
                 arma::vec& theta,
                 const arma::uvec& prioritized_features,
                 arma::vec& XTphi,
                 const arma::vec& XTtheta,
                 const double alpha,
                 const double theta_scale)
{
  // Updates phi via phi = (1-alpha)*phi + alpha*theta*theta_scale
  // Also updates ATphi
  // Requires values of ATtheta and ATphi to be current

  for (auto&& j : prioritized_features) {
    XTphi(j) = (1 - alpha) * XTphi(j) + alpha * theta_scale * XTtheta(j);
  }

  for (arma::uword i = 0; i < phi.n_elem; ++i)
    phi[i] = (1 - alpha) * phi(i) + alpha * theta_scale * theta(i);
}

// adopted from https://github.com/tbjohns/BlitzL1/blob/master/src/solver.cpp
double
Model::computeAlpha(const arma::uvec& prioritized_features,
                    const arma::vec& XTphi,
                    const arma::vec& XTtheta,
                    const double lambda,
                    const double theta_scale,
                    const arma::vec& X_norms_squared)
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

// adopted from https://github.com/tbjohns/BlitzL1/blob/master/src/solver.cpp
void
Model::prioritizeFeatures(arma::uvec& prioritized_features,
                          arma::vec& feature_priorities,
                          const arma::vec& XTphi,
                          const arma::vec& beta,
                          const arma::vec& X_norms_squared,
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
    prioritized_features(arma::sort_index(feature_priorities, "ascend"));
}

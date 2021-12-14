#include "model.h"

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

Model::~Model() = default;

void
Model::setLogHessianUpdateType(const std::string new_log_hessian_update_type){};

void
Model::updateLinearPredictor(const arma::mat& X)
{
  Xbeta = X * beta;
}

void
Model::updateLinearPredictor(const arma::sp_mat& X)
{
  Xbeta = X * beta;

  if (standardize)
    Xbeta -= arma::dot(beta, X_mean_scaled);
}

void
Model::updateLinearPredictor(const arma::mat& X, const arma::uvec& ind)
{
  Xbeta = X.cols(ind) * beta(ind);
}

void
Model::updateLinearPredictor(const arma::sp_mat& X, const arma::uvec& ind)
{
  Xbeta = X.cols(ind) * beta(ind);

  if (standardize)
    Xbeta -= arma::dot(beta(ind), X_mean_scaled(ind));
}

void
Model::updateCorrelation(const arma::mat& X)
{
  for (arma::uword j = 0; j < p; ++j) {
    c(j) = arma::dot(X.unsafe_col(j), residual);
  }
}

void
Model::updateCorrelation(const arma::sp_mat& X)
{
  for (arma::uword j = 0; j < p; ++j) {
    c(j) = arma::dot(X.col(j), residual);
  }

  if (standardize) {
    c -= X_mean_scaled * arma::accu(residual);
  }
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
    c(ind) -= X_mean_scaled(ind) * arma::accu(residual);
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

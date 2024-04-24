#include "linearAlgebra.h"

double
innerProduct(const arma::mat& X,
             const arma::uword j,
             const arma::vec& v,
             const arma::vec& X_offset,
             const bool standardize)
{
  return arma::dot(X.col(j), v);
}

double
innerProduct(const arma::sp_mat& X,
             const arma::uword j,
             const arma::vec& v,
             const arma::vec& X_offset,
             const bool standardize)
{
  double out = arma::dot(X.col(j), v);

  if (standardize)
    out -= X_offset(j) * arma::accu(v);

  return out;
}

double
weightedInnerProduct(const arma::mat& X,
                     const arma::uword j,
                     const arma::vec& v,
                     const arma::vec& weights,
                     const arma::vec& X_offset,
                     const bool standardize)
{
  return arma::dot(X.col(j), v % weights);
}

double
weightedInnerProduct(const arma::sp_mat& X,
                     const arma::uword j,
                     const arma::vec& v,
                     const arma::vec& weights,
                     const arma::vec& X_offset,
                     const bool standardize)
{
  double out = arma::dot(X.col(j) % weights, v);

  if (standardize) {
    out -= X_offset(j) * arma::dot(weights, v);
  }

  return out;
}

void
addScaledColumn(arma::vec& out,
                const arma::mat& X,
                const arma::uword j,
                const double a,
                const arma::vec& X_offset,
                const bool standardize)
{
  out += X.col(j) * a;
}

void
addScaledColumn(arma::vec& out,
                const arma::sp_mat& X,
                const arma::uword j,
                const double a,
                const arma::vec& X_offset,
                const bool standardize)
{
  out += a * X.col(j);

  if (standardize)
    out -= a * X_offset(j);
}

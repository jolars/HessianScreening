#include "updateCorrelation.h"

void
updateCorrelation(arma::vec& c,
                  const arma::vec& residual,
                  const arma::mat& X,
                  const arma::vec& offset,
                  const bool standardize)
{
  for (arma::uword j = 0; j < c.n_elem; ++j) {
    c(j) = arma::dot(X.unsafe_col(j), residual);
  }
}

void
updateCorrelation(arma::vec& c,
                  const arma::vec& residual,
                  const arma::sp_mat& X,
                  const arma::vec& offset,
                  const bool standardize)
{
  for (arma::uword j = 0; j < c.n_elem; ++j) {
    c(j) = arma::dot(X.col(j), residual);
  }

  if (standardize) {
    c -= offset * arma::accu(residual);
  }
}

void
updateCorrelation(arma::vec& c,
                  const arma::vec& residual,
                  const arma::mat& X,
                  const arma::uvec& ind,
                  const arma::vec& offset,
                  const bool standardize)
{
  for (auto&& j : ind) {
    c(j) = arma::dot(X.unsafe_col(j), residual);
  }
}

void
updateCorrelation(arma::vec& c,
                  const arma::vec& residual,
                  const arma::sp_mat& X,
                  const arma::uvec& ind,
                  const arma::vec& offset,
                  const bool standardize)
{
  for (auto&& j : ind) {
    c(j) = arma::dot(X.col(j), residual);
  }

  if (standardize) {
    c(ind) -= offset(ind) * arma::accu(residual);
  }
}

void
updateCorrelation(arma::vec& c,
                  const arma::vec& residual,
                  const arma::mat& X,
                  const arma::uword j,
                  const arma::vec& offset,
                  const bool standardize)
{
  c(j) = arma::dot(X.unsafe_col(j), residual);
}

void
updateCorrelation(arma::vec& c,
                  const arma::vec& residual,
                  const arma::sp_mat& X,
                  const arma::uword j,
                  const arma::vec& offset,
                  const bool standardize)
{
  c(j) = arma::dot(X.col(j), residual);

  if (standardize) {
    c(j) -= offset(j) * arma::accu(residual);
  }
}

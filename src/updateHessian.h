#pragma once

#include "model.h"
#include <RcppArmadillo.h>

template<typename T>
void
updateHessian(arma::mat& H,
              arma::mat& Hinv,
              arma::uvec& active_set,
              arma::uvec& active_set_prev,
              arma::uvec& active_perm,
              arma::uvec& active_perm_prev,
              const std::unique_ptr<Model>& model,
              const T& X,
              const arma::vec& X_offset,
              const bool standardize,
              const bool verify_hessian,
              const arma::uword verbosity)
{
  using namespace arma;

  const uword n = X.n_rows;

  const uvec deactivate = setDiff(active_set_prev, active_set);
  const uvec activate = setDiff(active_set, active_set_prev);

  if (!deactivate.is_empty()) {
    if (verbosity >= 1) {
      Rprintf("    dropping deactivated predictors for inverse (n = %i)\n",
              deactivate.n_elem);
    }

    std::vector<uword> keep_std, drop_std;

    for (uword i = 0; i < active_perm_prev.n_elem; ++i) {
      if (contains(active_perm, active_perm_prev(i))) {
        keep_std.emplace_back(i);
      } else {
        drop_std.emplace_back(i);
      }
    }

    const uvec keep(keep_std);
    const uvec drop(drop_std);

    const mat Hinv_kd = Hinv(keep, drop);

    Hinv = Hinv(keep, keep) -
           Hinv_kd * (solve(symmatu(Hinv(drop, drop)), Hinv_kd.t()));

    H.shed_cols(drop);
    H.shed_rows(drop);

    active_perm_prev.shed_rows(drop);
  }

  if (!activate.is_empty()) {
    if (verbosity >= 1) {
      Rprintf("    adding newly activated predictors to inverse (n = %i)\n",
              activate.n_elem);
    }

    mat D = model->hessian(X, activate, X_offset, standardize);
    mat B = model->hessianUpperRight(
      X, active_perm_prev, activate, X_offset, standardize);
    const mat S = symmatu(D - B.t() * Hinv * B);

    vec l;
    mat Q;
    eig_sym(l, Q, S);

    if (l.min() < 1e-4 * n) {
      D.diag() += 1e-4 * n;
      l += 1e-4 * n;
    }

    mat Sinv = Q * diagmat(1.0 / l) * Q.t();
    mat Hinv_B_Sinv = Hinv * B * Sinv;

    const uword N = H.n_cols;
    const uword M = D.n_cols;

    Hinv += Hinv_B_Sinv * B.t() * Hinv;
    Hinv.resize(N + M, N + M);
    Hinv.submat(0, N, size(N, M)) = -Hinv_B_Sinv;
    Hinv.submat(N, N, size(M, M)) = std::move(Sinv);
    Hinv = symmatu(Hinv);

    H.resize(N + M, N + M);
    H.submat(0, N, size(N, M)) = std::move(B);
    H.submat(N, N, size(M, M)) = std::move(D);
    H = symmatu(H);
  }

  if (verify_hessian) {
    double hess_inv_error = norm(H - H * Hinv * H, "inf");

    if (hess_inv_error >= 1e-2) {
      Rcpp::stop("inverse matrix computation is incorrect");
    }
  }
}

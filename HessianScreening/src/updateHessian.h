#pragma once

#include <RcppArmadillo.h>

#include "model.h"

using namespace arma;

template<typename T>
void
updateHessian(mat& H,
              mat& Hinv,
              uvec& active_set,
              uvec& active_prev_set,
              const std::unique_ptr<Model>& model,
              const T& X,
              const bool verify_hessian,
              const bool approx_hessian,
              const uword verbosity,
              const bool reset_hessian)
{
  const uword n = X.n_rows;

  uvec deactivate = safeSetDiff(active_prev_set, active_set);
  uvec activate = safeSetDiff(active_set, active_prev_set);

  if (!deactivate.is_empty() && !reset_hessian) {
    if (verbosity >= 1) {
      Rprintf("    dropping deactivated predictors for inverse (n = %i)\n",
              deactivate.n_elem);
    }

    std::vector<uword> keep_std, drop_std;

    for (uword i = 0; i < active_prev_set.n_elem; ++i) {
      if (contains(active_set, active_prev_set(i))) {
        keep_std.emplace_back(i);
      } else {
        drop_std.emplace_back(i);
      }
    }

    const uvec keep = conv_to<uvec>::from(keep_std);
    const uvec drop = conv_to<uvec>::from(drop_std);

    const mat Hinv_kd = Hinv(keep, drop);
    const mat Hinv_kk = Hinv(keep, keep);
    const mat Hinv_dd = Hinv(drop, drop);

    Hinv = Hinv_kk - Hinv_kd * (solve(symmatu(Hinv_dd), Hinv_kd.t()));

    H.shed_cols(drop);
    H.shed_rows(drop);

    active_prev_set = safeSetIntersect(active_prev_set, active_set);
  }

  if (!activate.is_empty()) {
    if (verbosity >= 1) {
      Rprintf("    adding newly activated predictors to inverse (n = %i)\n",
              activate.n_elem);
    }

    mat D = model->hessian(X, activate);
    const mat B = model->hessianUpperRight(X, active_prev_set, activate);
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

    const uword H_n = H.n_rows;
    const uword H_p = H.n_cols;

    H.resize(H.n_rows + D.n_rows, H.n_cols + D.n_cols);
    H.submat(0, H_n, size(B)) = B;
    H.submat(H_n, H_p, size(D)) = D;
    H = symmatu(H);
    mat H_00(size(Hinv));
    H_00 = Hinv_B_Sinv * B.t() * Hinv + Hinv;
    Hinv.set_size(H.n_rows, H.n_cols);
    Hinv.submat(0, 0, size(H_00)) = H_00;
    Hinv.submat(0, H_n, size(B)) = -Hinv_B_Sinv;
    Hinv.submat(H_n, H_p, size(D)) = Sinv;
    Hinv = symmatu(Hinv);
  }

  if (reset_hessian) {
    H = X.cols(active_set).t() * X.cols(active_set);
    H.diag() += 1e-4 * n;
    mat Q;
    vec l;
    eig_sym(l, Q, symmatu(H));
    Hinv = Q * diagmat(1 / l) * Q.t();
  }

  if (verify_hessian) {
    double hess_inv_error = norm(H - H * Hinv * H, "inf");

    if (hess_inv_error >= 1e-2) {
      Rcpp::stop("inverse matrix computation is incorrect");
    }
  }
}

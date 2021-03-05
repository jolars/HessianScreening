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
  const uword p = X.n_cols;

  uvec deactivate = setDiff(active_prev_set, active_set);
  uvec activate = setDiff(active_set, active_prev_set);

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

    uvec keep = conv_to<uvec>::from(keep_std);
    uvec drop = conv_to<uvec>::from(drop_std);

    mat Hinv_kd = Hinv(keep, drop);
    mat Hinv_kk = Hinv(keep, keep);
    mat Hinv_dd = Hinv(drop, drop);

    Hinv = symmatu(Hinv_kk - Hinv_kd * (solve(symmatu(Hinv_dd), Hinv_kd.t())));
    H = symmatu(H(keep, keep));

    active_prev_set = setIntersect(active_prev_set, active_set);
  }

  if (!activate.is_empty()) {
    if (verbosity >= 1) {
      Rprintf("    adding newly activated predictors to inverse (n = %i)\n",
              activate.n_elem);
    }

    mat D = model->hessian(X, activate);
    mat B = model->hessianUpperRight(X, active_prev_set, activate);

    mat Hinv_B = Hinv * B;

    mat S = D - B.t() * Hinv_B;

    vec l;
    mat Q;
    eig_sym(l, Q, symmatu(S));

    if (l.min() < 1e-4 * n) {
      D.diag() += 1e-4 * n;
      l += 1e-4 * n;
    }

    mat Sinv = Q * diagmat(1.0 / l) * Q.t();
    mat Hinv_B_Sinv = Hinv_B * Sinv;

    H = symmatu(join_vert(join_horiz(H, B), join_horiz(B.t(), D)));

    uword n_old = active_prev_set.n_elem;

    Hinv =
      join_vert(join_horiz(Hinv_B_Sinv * B.t() * Hinv + Hinv, -Hinv_B_Sinv),
                join_horiz(-Hinv_B_Sinv.t(), Sinv));
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

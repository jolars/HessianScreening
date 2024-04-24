#pragma once

#include <RcppArmadillo.h>

inline bool
exists_sphere_twohalfspace(const double rsq,
                           const double a,
                           const double b,
                           const double vTv,
                           const double wTw,
                           const double vTw)
{
  // candidate of argmin min_{||t||}_2^2 under t^Tv >= a and t^Tw >= b
  // = 0, a/v^Tv*v, b/w^Tw*w, (v w) ((v w)^T(v w))^(-1) (a b)^T

  return ((a <= 0 && b <= 0) ||
          (0 < vTv && a * a / vTv <= rsq && b <= a / vTv * vTw) ||
          (0 < wTw && b * b / wTw <= rsq && a <= b / wTw * vTw) ||
          (0 < (vTv * wTw - vTw * vTw) &&
           (a * a * wTw + b * b * vTv - 2.0 * a * b * vTw) /
               (vTv * wTw - vTw * vTw) <=
             rsq));
}

inline void
screening_dsasvi_lasso_lincombtheta(const arma::uword d,
                                    arma::uvec& screened,
                                    const arma::vec& XTXbeta,
                                    const arma::vec& XTy,
                                    const arma::vec& l2s,
                                    const double l1beta,
                                    const double l2Xbeta,
                                    const double l2y,
                                    const double yTXbeta,
                                    const double coef_Xbeta,
                                    const double coef_y)
{
  // theta = coef_Xbeta * Xbeta + coef_y * y

  // exists or not z which    {||z-(theta-y)/2||}_2^2 <= {||(theta+y)/2||}_2^2
  // and z^T(-Xbeta) <= g(beta) and z^T(-x_j) >= 1 = exists or not z which
  // {||z||}_2^2 <= {||(theta+y)/2||}_2^2 and z^TXbeta >= -g(beta) -
  // ((theta-y)/2)^TXbeta and z^T(-x_j) >= 1 + ((theta-y)/2)^Tx_j
  const double DSASVI_SAFETY_EPS = 0.0000001;

  double rsq = (coef_Xbeta * coef_Xbeta * l2Xbeta * l2Xbeta +
                (coef_y + 1.0) * (coef_y + 1.0) * l2y * l2y +
                2.0 * coef_Xbeta * (coef_y + 1.0) * yTXbeta) /
               4.0;
  double oTXbeta =
    (coef_Xbeta * l2Xbeta * l2Xbeta + (coef_y - 1.0) * yTXbeta) / 2.0;

  for (arma::uword j = 0; j < d; j++) {
    if (!screened(j))
      continue;

    double oTxj = (coef_Xbeta * XTXbeta(j) + (coef_y - 1) * XTy(j)) / 2.0;
    bool eliminate = !exists_sphere_twohalfspace(
                       rsq,
                       -l1beta - oTXbeta - DSASVI_SAFETY_EPS * l2Xbeta,
                       1.0 + oTxj - DSASVI_SAFETY_EPS * l2s(j),
                       l2Xbeta * l2Xbeta,
                       l2s(j) * l2s(j),
                       -XTXbeta(j)) &&
                     !exists_sphere_twohalfspace(
                       rsq,
                       -l1beta - oTXbeta - DSASVI_SAFETY_EPS * l2Xbeta,
                       1.0 - oTxj - DSASVI_SAFETY_EPS * l2s(j),
                       l2Xbeta * l2Xbeta,
                       l2s(j) * l2s(j),
                       XTXbeta(j));

    if (eliminate) {
      screened(j) = false;
    }
  }
}

/*
void
screening_dsasvi_lasso_lincombtheta(const int d,
                                    bool* eliminated,
                                    const double* XTXbeta,
                                    const double* XTy,
                                    const double* l2s,
                                    const double l1beta,
                                    const double l2Xbeta,
                                    const double l2y,
                                    const double yTXbeta,
                                    const double coef_Xbeta,
                                    const double coef_y)
{
  // theta = coef_Xbeta * Xbeta + coef_y * y

  // exists or not z which    {||z-(theta-y)/2||}_2^2 <= {||(theta+y)/2||}_2^2
  // and z^T(-Xbeta) <= g(beta) and z^T(-x_j) >= 1 = exists or not z which
  // {||z||}_2^2 <= {||(theta+y)/2||}_2^2 and z^TXbeta >= -g(beta) -
  // ((theta-y)/2)^TXbeta and z^T(-x_j) >= 1 + ((theta-y)/2)^Tx_j

  double rsq = (coef_Xbeta * coef_Xbeta * l2Xbeta * l2Xbeta +
                (coef_y + 1.0) * (coef_y + 1.0) * l2y * l2y +
                2.0 * coef_Xbeta * (coef_y + 1.0) * yTXbeta) /
               4.0;
  double oTXbeta =
    (coef_Xbeta * l2Xbeta * l2Xbeta + (coef_y - 1.0) * yTXbeta) / 2.0;
  for (int j = 0; j < d; j++) {
    if (*(eliminated + j)) {
      continue;
    }
    double oTxj =
      (coef_Xbeta * *(XTXbeta + j) + (coef_y - 1) * *(XTy + j)) / 2.0;
    *(eliminated + j) = !exists_sphere_twohalfspace(
                          rsq,
                          -l1beta - oTXbeta - DSASVI_SAFETY_EPS * l2Xbeta,
                          1.0 + oTxj - DSASVI_SAFETY_EPS * *(l2s + j),
                          l2Xbeta * l2Xbeta,
                          *(l2s + j) * *(l2s + j),
                          -*(XTXbeta + j)) &&
                        !exists_sphere_twohalfspace(
                          rsq,
                          -l1beta - oTXbeta - DSASVI_SAFETY_EPS * l2Xbeta,
                          1.0 - oTxj - DSASVI_SAFETY_EPS * *(l2s + j),
                          l2Xbeta * l2Xbeta,
                          *(l2s + j) * *(l2s + j),
                          *(XTXbeta + j));
  }
} */

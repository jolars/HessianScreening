#include <RcppArmadillo.h>

using namespace arma;

// [[Rcpp::export]]
void
blockMatrixConstruction(int method = 0, unsigned n = 9000, unsigned m = 50)
{
  mat H(n, n, fill::zeros);
  mat A(n, n, fill::zeros);
  mat B(n, m, fill::zeros);
  mat D(m, m, fill::zeros);

  if (method == 0) {
    H.set_size(n + m, n + m);
    H.submat(0, 0, size(A)) = std::move(A);
    H.submat(0, n, size(B)) = std::move(B);
    H.submat(n, n, size(D)) = std::move(D);
    H = symmatu(H);
  } else if (method == 1) {
    mat C(m, n);
    H = join_vert(join_horiz(A, B), join_horiz(C, D));
    H = symmatu(H);
  } else {
    H = std::move(A);
    H.resize(n + m, n + m);
    H.submat(0, n, size(B)) = std::move(B);
    H.submat(n, n, size(D)) = std::move(D);
    H = symmatu(H);
  }
}

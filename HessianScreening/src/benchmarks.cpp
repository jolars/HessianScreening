#include <RcppArmadillo.h>
#include <omp.h>

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

// [[Rcpp::export]]
void
denseInnerProduct(int method = 0,
                  unsigned n = 100,
                  unsigned p = 1000,
                  unsigned m = 50)
{
  mat A(n, p, fill::randu);
  vec b(n, fill::randu);

  uvec ind = sort(randperm(p, m));

  if (method == 0) {
    vec Atb = A.cols(ind).t() * b;
  } else if (method == 1) {
    vec Atb(p);

    for (auto&& j : ind) {
      Atb(j) = dot(A.unsafe_col(j), b);
    }
  } else if (method == 2) {
    vec Atb(p);

    for (auto&& j : ind) {
      Atb(j) = dot(A.unsafe_col(j), b);
    }
  }
}

// [[Rcpp::export]]
void
sparseInnerProduct(int method = 0,
                   unsigned n = 100,
                   unsigned p = 1000,
                   unsigned m = 50,
                   double density = 0.1,
                   int n_threads = 4)
{
  sp_mat A = sprandu<sp_mat>(n, p, density);
  vec b(n, fill::randu);

  uvec ind = sort(randperm(p, m));

  wall_clock timer;
  timer.tic();

  if (method == 0) {
    vec Atb = A.cols(ind).t() * b;
  } else if (method == 1) {
    vec Atb(p);

    for (auto&& j : ind) {
      Atb(j) = dot(A.col(j), b);
    }
  } else if (method == 2) {
    vec Atb(p);

#pragma omp parallel for num_threads(n_threads)
    for (uword j = 0; j < ind.n_elem; ++j) {
      Atb(ind(j)) = dot(A.col(ind(j)), b);
    }
  }
}

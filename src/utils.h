#pragma once

#include <RcppArmadillo.h>

inline double
squaredNorm(const arma::vec& x)
{
  return std::pow(arma::norm(x), 2);
}

template<typename T>
inline int
signum(T val)
{
  return (T(0) < val) - (val < T(0));
}

template<typename T, typename S>
inline bool
contains(const T& x, const S& what)
{
  return std::find(x.begin(), x.end(), what) != x.end();
}

inline arma::uvec
setUnion(const arma::uvec& a, const arma::uvec& b)
{
  std::vector<unsigned> out;
  out.reserve(a.n_elem + b.n_elem);

  std::set_union(
    a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));

  out.shrink_to_fit();

  return arma::conv_to<arma::uvec>::from(out);
}

inline arma::uvec
setDiff(const arma::uvec& a, const arma::uvec& b)
{
  std::vector<arma::uword> out;
  out.reserve(a.n_elem);

  std::set_difference(
    a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));

  out.shrink_to_fit();

  return arma::conv_to<arma::uvec>::from(out);
}

inline arma::uvec
setIntersect(const arma::uvec& a, const arma::uvec& b)
{
  std::vector<arma::uword> out;
  out.reserve(std::min(a.n_elem, b.n_elem));

  std::set_intersection(
    a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));

  out.shrink_to_fit();

  return arma::conv_to<arma::uvec>::from(out);
}

// set intersection that retains permutation in `a`
inline arma::uvec
safeSetIntersect(const arma::uvec& a, const arma::uvec& b)
{
  std::vector<arma::uword> out;
  out.reserve(std::min(a.n_elem, b.n_elem));

  for (auto&& a_i : a) {
    if (contains(b, a_i)) {
      out.emplace_back(a_i);
    }
  }

  out.shrink_to_fit();

  return arma::conv_to<arma::uvec>::from(out);
}

// set difference that retains permutation in `a`
inline arma::uvec
safeSetDiff(const arma::uvec& a, const arma::uvec& b)
{
  std::vector<arma::uword> out;
  out.reserve(a.n_elem);

  for (auto&& a_i : a) {
    if (!contains(b, a_i)) {
      out.emplace_back(a_i);
    }
  }

  out.shrink_to_fit();

  return arma::conv_to<arma::uvec>::from(out);
}

inline arma::vec
matTransposeMultiply(const arma::mat& A,
                     const arma::vec& b,
                     const arma::vec& offset,
                     const bool standardize)
{
  return A.t() * b;
}

inline arma::vec
matTransposeMultiply(const arma::sp_mat& A,
                     const arma::vec& b,
                     const arma::vec& offset,
                     const bool standardize)
{
  arma::vec Atb = A.t() * b;

  if (standardize) {
    Atb -= offset * arma::accu(b);
  }

  return Atb;
}

inline arma::vec
matTransposeMultiply(const arma::mat& A,
                     const arma::vec& b,
                     const arma::uvec& ind,
                     const arma::vec& offset,
                     const bool standardize)
{
  return A.cols(ind).t() * b;
}

inline arma::vec
matTransposeMultiply(const arma::sp_mat& A,
                     const arma::vec& b,
                     const arma::uvec& ind,
                     const arma::vec& offset,
                     const bool standardize)
{
  arma::vec Atb = A.cols(ind).t() * b;

  if (standardize) {
    Atb -= offset * arma::accu(b);
  }

  return Atb;
}

inline double
getSparsity(const arma::mat& X)
{
  return 0.0;
}

inline double
getSparsity(const arma::sp_mat& X)
{
  return 1.0 - static_cast<double>(X.n_nonzero) / X.n_elem;
}

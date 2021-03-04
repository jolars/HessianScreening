#pragma once

#include <RcppArmadillo.h>

using namespace arma;

inline double
squaredNorm(const vec& x)
{
  return std::pow(norm(x), 2);
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

inline uvec
setUnion(const uvec& a, const uvec& b)
{
  std::vector<uword> out = conv_to<std::vector<uword>>::from(a);

  for (auto&& b_i : b) {
    if (!contains(out, b_i)) {
      out.emplace_back(b_i);
    }
  }

  return conv_to<uvec>::from(out);
}

inline uvec
setDiff(const uvec& a, const uvec& b)
{
  std::vector<uword> out;

  for (auto&& a_i : a) {
    if (!contains(b, a_i)) {
      out.emplace_back(a_i);
    }
  }

  return conv_to<uvec>::from(out);
}

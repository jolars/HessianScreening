#include <RcppArmadillo.h>
#include "lassoPath.hpp"

// [[Rcpp::export]]
Rcpp::List
lassoPathDense(arma::mat X,
               arma::vec y,
               const std::string family,
               const bool standardize,
               const std::string screening_type,
               const bool hessian_warm_starts,
               const bool gap_safe_active_start,
               std::string log_hessian_update_type,
               const arma::uword log_hessian_auto_update_freq,
               const arma::uword path_length,
               const arma::uword maxit,
               const double tol_infeas,
               const double tol_gap,
               const double gamma,
               const bool verify_hessian,
               const bool force_kkt_check,
               const int line_search,
               const arma::uword verbosity)
{
  return lassoPath(X,
                   y,
                   family,
                   standardize,
                   screening_type,
                   hessian_warm_starts,
                   gap_safe_active_start,
                   log_hessian_update_type,
                   log_hessian_auto_update_freq,
                   path_length,
                   maxit,
                   tol_infeas,
                   tol_gap,
                   gamma,
                   verify_hessian,
                   force_kkt_check,
                   line_search,
                   verbosity);
}

// [[Rcpp::export]]
Rcpp::List
lassoPathSparse(arma::sp_mat X,
                arma::vec y,
                const std::string family,
                const bool standardize,
                const std::string screening_type,
                const bool hessian_warm_starts,
                const bool gap_safe_active_start,
                std::string log_hessian_update_type,
                const arma::uword log_hessian_auto_update_freq,
                const arma::uword path_length,
                const arma::uword maxit,
                const double tol_infeas,
                const double tol_gap,
                const double gamma,
                const bool verify_hessian,
                const bool force_kkt_check,
                const int line_search,
                const arma::uword verbosity)
{
  return lassoPath(X,
                   y,
                   family,
                   standardize,
                   screening_type,
                   hessian_warm_starts,
                   gap_safe_active_start,
                   log_hessian_update_type,
                   log_hessian_auto_update_freq,
                   path_length,
                   maxit,
                   tol_infeas,
                   tol_gap,
                   gamma,
                   verify_hessian,
                   force_kkt_check,
                   line_search,
                   verbosity);
}

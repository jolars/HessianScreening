#include <RcppArmadillo.h>

#include <regex>
#include <sstream>
#include <fstream>
#include <string>

using namespace arma;
using namespace Rcpp;

Rcpp::List
readLIBSVM(const std::string path)
{
  std::ifstream t(path.c_str());
  std::stringstream ss;
  ss << t.rdbuf();

  std::vector<double> y;
  std::vector<double> values;
  std::vector<int> row_ind;
  std::vector<int> col_ind;

  std::string to;

  int i = 0;

  while (std::getline(ss, to, '\n')) {

    std::regex rgx("\\s+");

    std::sregex_token_iterator it(std::begin(to), std::end(to), rgx, -1);
    std::sregex_token_iterator it_end;

    // first element is response
    y.emplace_back(std::stod(*it));
    ++it;

    for (; it != it_end; ++it) {
      std::string b = *it;
      std::size_t pos = b.rfind(':');

      double val = std::stod(b.substr(pos + 1, b.length()));

      if (val != 0) {
        row_ind.emplace_back(i);
        col_ind.emplace_back(std::stoi(b.substr(0, pos)) - 1);
        values.emplace_back(val);
      }
    }

    ++i;
  }

  umat locations = join_vert(conv_to<urowvec>::from(row_ind),
                             conv_to<urowvec>::from(col_ind));

  vec arma_values = conv_to<vec>::from(values);

  uword n_rows = i;
  uword n_cols = locations.row(1).max() + 1;

  sp_mat X(locations, arma_values, n_rows, n_cols);

  return Rcpp::List::create(
    Rcpp::Named("y") = Rcpp::wrap(y),
    Rcpp::Named("X") = Rcpp::wrap(X)
  );
}

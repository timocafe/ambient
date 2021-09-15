

namespace dl {

// notation - coursera
// Neural Networks and Deep Learning
// week - 4

namespace fs = std::filesystem;
namespace am = ambient;
namespace bg = boost::gil;
using matrix = am::tiles<am::matrix<float>>;

struct layer {

  layer(const int row, const int col) : row_(row), col_(col) {
    w_ = matrix(row, col);
    b_ = matrix(row, 1, 1.);
    generate(w_);
  }

  auto linear_forward(const matrix &a) { return w_ * a + b_; }

  matrix w_;
  matrix b_;
  int row_;
  int col_;
};

struct network {

  network(const std::vector<int> &layer_dims) {
    v_.reserve(1024);
    for (int i = 1; i < layer_dims.size(); ++i) {
      const int row = layer_dims[i];
      const int col = layer_dims[i - 1];
      v_.push_back(layer(row, col));
    }
  }

  std::vector<layer> v_;
};

} // namespace dl

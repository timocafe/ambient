#include <algorithm>
#include <filesystem>
#include <iostream>
#include <list>
#include <random>
#include <string>

#include <boost/gil.hpp>
#include <boost/gil/extension/io/jpeg.hpp>
#include <boost/gil/extension/numeric/resample.hpp>
#include <boost/gil/extension/numeric/sampler.hpp>

#include "ambient/ambient.hpp"
#include "ambient/container/numeric/matrix.hpp"

#include "neural_network.h"

namespace fs = std::filesystem;
namespace am = ambient;
namespace bg = boost::gil;
using matrix = am::tiles<am::matrix<float>>;

const static int SIZE = 128;

struct reader {
  typedef typename boost::gil::rgb32f_image_t::value_type value_type;

  reader(const fs::path &origin) {
    std::random_device rd;
    std::mt19937 generator(rd());
    files_.reserve(1024);
    for (const auto &entry : fs::directory_iterator(origin))
      files_.push_back(entry.path());
    std::shuffle(files_.begin(), files_.end(), generator);
  }

  const auto size() { return files_.size(); }

  const std::vector<fs::path> &files() { return files_; }

  std::vector<float> raw_data(const std::string &filename) {
    const auto &p = fs::path(filename);
    bg::rgb8_image_t image;
    bg::read_and_convert_image(filename, image, bg::jpeg_tag());
    bg::rgb8_image_t square(SIZE, SIZE);
    bg::resize_view(const_view(image), view(square), bg::bilinear_sampler());
    const auto view = bg::view(square);
    auto *data = bg::interleaved_view_get_raw_data(view);
    uint64_t length = view.width() * view.height() * view.num_channels();
    return std::vector<float>(data, data + length);
  }

  std::vector<fs::path> files_;
};

void fillup(reader r, matrix &m, const std::size_t size) {
  assert(size < r.size() && " too ambitious !\n");
  for (int i = 0; i < size; ++i) {
    const auto &fname = r.files()[i];
    const std::vector<float> &v = r.raw_data(fname.string());
    fill_col(m, v, i);
  }
}

int main(int argc, char *argv[]) {
  fs::path origin("/Users/tewart/Documents/train");
  reader r(origin);

  std::size_t nrows = SIZE * SIZE * 3;
  std::size_t ncols = 256;
  matrix X(nrows, ncols);
  fillup(r, X, ncols);

  dl::layer l1(ncols, nrows);
  auto a = l1.linear_forward(X);
  ambient::sync();

  std::cout << " hello \n";
}

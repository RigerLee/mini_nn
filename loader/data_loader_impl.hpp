/**
 * @file data_loader_impl.hpp
 * @brief data_loader_impl.hpp
 * @details head file
 * @mainpage mini_nn
 * @author RuiJian Li, YiFan Cao, YanPeng Hu
 * @email lirj@shanghaitech.edu.cn,
 * caoyf@shanghaitech.edu.cn,huyp@shanghaitech.edu.cn
 * @version 1.6.0
 * @date 2019-05-30
 */

#define my_max(a, b) (((a) > (b)) ? (a) : (b))
#define my_min(a, b) (((a) < (b)) ? (a) : (b))

template <typename T>
Dataset<T>::Dataset(bool shuffle) : shuffle_(shuffle) {}

template <typename T>
void Dataset<T>::MNIST(const std::string& path) {
  std::string train_image = path + "train-images-idx3-ubyte";
  std::string train_label = path + "train-labels-idx1-ubyte";
  std::string test_image = path + "t10k-images-idx3-ubyte";
  std::string test_label = path + "t10k-labels-idx1-ubyte";
  // load train and test when init
  read_bin_images(train_image, train_data_);
  read_bin_images(test_image, test_data_);
  read_bin_labels(train_label, train_label_);
  read_bin_labels(test_label, test_label_);
  std::cout << "train: ";
  cout_shape(train_data_);
  std::cout << "test: ";
  cout_shape(test_data_);
}

template <typename T>
std::vector<std::pair<xt::xarray<T>, xt::xarray<T>>> Dataset<T>::loader(
  const std::string& mode, int batch_size) {
  // load train or test
  auto data = mode == "train" ? train_data_ : test_data_;
  auto label = mode == "train" ? train_label_ : test_label_;
  xt::xarray<size_t> indexs = xt::arange(0, (int)data.shape(0));
  if (shuffle_) {
    // Set srand
    struct timeval t;
    gettimeofday(&t, NULL);
    xt::random::seed(t.tv_usec);
    xt::random::shuffle(indexs);
  }
  // ceiling divide
  int num_batches = (data.shape(0) + batch_size - 1) / batch_size;
  // generate batch and store in out
  std::vector<std::pair<xt::xarray<T>, xt::xarray<T>>> out;
  for (int i = 0; i < num_batches; ++i) {
    auto rows = xt::view(
      indexs, xt::range(i * batch_size,
                        my_min((int)data.shape(0), (i + 1) * batch_size)));
    // stupied library to index array...  rows and idx are identity...
    // has use flatten_indices to convert array to index type
    auto idx = xt::flatten_indices(xt::argwhere(rows >= 0));
    auto data_temp =
      Matrix(xt::view(data, xt::keep(idx), xt::all(), xt::all(), xt::all()));
    auto label_temp = Matrix(xt::view(label, xt::keep(idx)));
    // add data and label
    out.push_back(std::make_pair(data_temp, label_temp));
  }
  return out;
}

int read_int(char* in) {
  int out;
  char* p = (char*)&out;
  // deal with smallendian
  p[0] = in[3];
  p[1] = in[2];
  p[2] = in[1];
  p[3] = in[0];
  return out;
}

template <typename T>
void Dataset<T>::read_bin_images(const std::string& image_file,
                                 xt::xarray<T>& data) {
  FILE* fp = fopen(image_file.c_str(), "rb");
  std::vector<uint8_t> images;
  // int* num = [magic number, num_images, rows, cols]
  char smallendian[4];
  auto ret = fread(smallendian, sizeof(char), 4, fp);
  if (ret != 4) throw std::runtime_error("MNIST: fread failed");
  // check magic number
  if (read_int(smallendian) != 2051)
    throw std::runtime_error("MNIST: fread not bin image file");
  // check number
  ret = fread(smallendian, sizeof(char), 4, fp);
  int num_images = read_int(smallendian);
  // check shape
  ret = fread(smallendian, sizeof(char), 4, fp);
  if (read_int(smallendian) != 28)
    throw std::runtime_error("MNIST: fread not bin image file");
  ret = fread(smallendian, sizeof(char), 4, fp);
  if (read_int(smallendian) != 28)
    throw std::runtime_error("MNIST: fread not bin image file");
  images.resize(num_images * 28 * 28);
  ret = fread(images.data(), 1, num_images * 28 * 28, fp);
  // first to uint8_t xarray
  const xt::xarray<uint8_t> data_temp =
    xt::adapt(images, Shape({(long unsigned int)num_images, 1, 28, 28}));
  // cast to T, map to 0-1
  data = xt::cast<T>(data_temp) / 255;
  fclose(fp);
  // std::cout << "Read images " << image_file << " done.\n";
}

template <typename T>
void Dataset<T>::read_bin_labels(const std::string& label_file,
                                 xt::xarray<T>& label) {
  FILE* fp = fopen(label_file.c_str(), "rb");
  std::vector<uint8_t> labels;
  // int* num = [magic number, num_images, rows, cols]
  char smallendian[4];
  auto ret = fread(smallendian, sizeof(char), 4, fp);
  if (ret != 4) throw std::runtime_error("MNIST: fread failed");
  // check magic number
  if (read_int(smallendian) != 2049)
    throw std::runtime_error("MNIST: fread not bin label file");
  // check number
  ret = fread(smallendian, sizeof(char), 4, fp);
  int num_labels = read_int(smallendian);

  labels.resize(num_labels);
  ret = fread(labels.data(), 1, num_labels, fp);
  // first to uint8_t xarray
  const xt::xarray<uint8_t> label_temp =
    xt::adapt(labels, Shape({(long unsigned int)num_labels}));
  // cast to T
  label = xt::cast<T>(label_temp);
  fclose(fp);
  // std::cout << "Read labels " << label_file << " done.\n";
}

template <typename T>
void Dataset<T>::normalize(const xt::xarray<T>& mean,
                           const xt::xarray<T>& stdev) {
  // mean and std: 1D array
  if (mean.shape(0) != stdev.shape(0))
    throw std::runtime_error("normalize: mean and stdev have different shape");
  if (mean.shape(0) != train_data_.shape(1))
    throw std::runtime_error(
      "normalize: mean/stdev and images have different channel");
  // result = (in - mean) / std
  train_data_ =
    (train_data_ -
     xt::view(mean, xt::newaxis(), xt::all(), xt::newaxis(), xt::newaxis())) /
    xt::view(stdev, xt::newaxis(), xt::all(), xt::newaxis(), xt::newaxis());

  test_data_ =
    (test_data_ -
     xt::view(mean, xt::newaxis(), xt::all(), xt::newaxis(), xt::newaxis())) /
    xt::view(stdev, xt::newaxis(), xt::all(), xt::newaxis(), xt::newaxis());
}

template <typename T>
xt::xarray<T> load_images(const std::vector<std::string>& paths,
                          const xt::xarray<T>& mean,
                          const xt::xarray<T>& std) {
  typename xt::xarray<T>::shape_type rshape = {0};
  xt::xarray<T> result(rshape);
  bool not_reshaped = true;
  for (auto& path : paths) {
    xt::xarray<T> image = xt::transpose(xt::load_image(path), {2, 0, 1});
    normalize(mean, std);
    auto shape = image.shape();
    size_t C = shape[0];
    size_t H = shape[1];
    size_t W = shape[2];
    image.reshape({1, C, H, W});
    if (not_reshaped) {
      result.reshape({0, C, H, W});
      not_reshaped = false;
    }
    result = xt::concatenate(xt::xtuple(result, image));
  }
  return result;
}

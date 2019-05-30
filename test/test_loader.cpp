/**
 * @file test_loader.cpp
 * @author RuiJian Li(lirj@shanghaitech.edu.cn), YiFan Cao(caoyf@shanghaitech.edu.cn), YanPeng Hu(huyp@shanghaitech.edu.cn)
 * @brief to test the loader  whether worked out or not
 * @version 1.6.0
 * @date 2019-05-30
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#include "data_loader.hpp"

#include "model_loader.hpp"
#include "xtensor/xsort.hpp"

using namespace std;
/**
 * @brief load the model
 *
 * @param argc
 * @param argv
 * @return int
 */
int main(int argc, char** argv) {
  // std::vector<std::string> v {"a.jpg", "b.jpg"};
  // auto result = load_images<double>(v, {0, 0, 0}, {1, 2, 4});
  // std::cout << result << std::endl;
  Dataset<double> loader;
  loader.MNIST("./data/");
  loader.normalize({0.1307}, {0.3081});
  auto train = loader.loader("train", 10);
  cout << train.size() << endl;

  Network<float> net;
  // auto input = xt::linspace<float>(-1., 1., 784).reshape({1, 1, 28, 28});
  load_model(net, "./model/layer.txt");
  auto temp = net.predict(train[0].first);
  temp = xt::argmax(temp, 1);
  cout << temp << endl;
  cout << train[0].second << endl;

  return 0;
}

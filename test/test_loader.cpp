/**
 * @file test_loader.cpp
 * @author RuiJian Li(lirj@shanghaitech.edu.cn), YiFan
 * Cao(caoyf@shanghaitech.edu.cn), YanPeng Hu(huyp@shanghaitech.edu.cn)
 * @brief to test the loader  whether worked out or not
 * @version 1.6.0
 * @date 2019-05-30
 *
 * @copyright Copyright (c) 2019
 *
 */
#include "data_loader.hpp"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include "model_loader.hpp"

using namespace std;
/**
 * @brief load the model
 *
 * @param argc
 * @param argv
 * @return int
 */
int main(int argc, char** argv) {
  Dataset<float> loader;
  loader.MNIST("./data/");
  loader.normalize({0.1307}, {0.3081});
  auto test = loader.loader("test", 1000);

  int correct = 0;
  int total = 0;
  Network<float> net;
  load_model(net, "./model/layer.txt");
  auto t_start = std::chrono::system_clock::now();
  for (auto& t : test) {
    auto temp = xt::argmax(net.predict(t.first), 1);
    auto target = t.second;
    for (size_t i = 0; i < temp.size(); ++i) {
      if (temp(i) - target(i) < 1e-6) ++correct;
    }
    total += temp.size();
    cout << "Acc: " << (float)correct / total << endl;
  }
  auto t_end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = t_end - t_start;
  cout << "Total time: " << elapsed_seconds.count() << " s" << endl;

  // auto acc = xt::sum(, {0});
  // acc /= test[0].second.size();

  return 0;
}

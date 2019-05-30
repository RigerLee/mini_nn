#include "data_loader.hpp"
#include "network.hpp"
#include "utils.hpp"

int main() {
  Dataset<double> loader;
  loader.MNIST("./data/");
  loader.normalize({0.1307}, {0.3081});
  auto train = loader.loader("train", 640);
  auto test = loader.loader("test", 1000);

  std::cout << "start" << std::endl;

  Network<double> net;

  net.set_optimizer(new SGD<double>(0.001, 0.5));

  net << new Conv2d<double>(1, 20, 5, 1) << new ReLU<double>()
      << new MaxPool2d<double>(2, 2) << new Conv2d<double>(20, 50, 5, 1)
      << new ReLU<double>() << new MaxPool2d<double>(2, 2)
      << new Linear<double>(4 * 4 * 50, 500) << new ReLU<double>()
      << new Linear<double>(500, 10) << new Loss<double>(CROSS_ENTROPY);
  //  net << new Linear<double>(28 * 28, 500) << new ReLU<double>()
  //      << new Linear<double>(500, 256) << new ReLU<double>()
  //      << new Linear<double>(256, 10) << new Loss<double>(CROSS_ENTROPY);
  for (int i = 0; i < 10; ++i) {
    for (size_t j = 0; j < train.size(); ++j) {
      auto& pair = train[j];
      auto& data = pair.first;
      auto& target = pair.second;
      auto loss = net.forward(data, target);
      net.backward();
      std::cout << "Epoch " << i << " batch " << j << " loss " << loss
                << std::endl;
    }
  }

  return 0;
}

#include "data_loader.hpp"

int main(int argc, char** argv) {
  std::vector<std::string> v {"a.jpg", "b.jpg"};
  auto result = load_images<int>(v);
  std::cout << result << std::endl;
  return 0;
}

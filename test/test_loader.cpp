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
/**
 * @brief 
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char** argv) {
  std::vector<std::string> v {"a.jpg", "b.jpg"};
  auto result = load_images<double>(v, {0, 0, 0}, {1, 2, 4});
  std::cout << result << std::endl;
  return 0;
}

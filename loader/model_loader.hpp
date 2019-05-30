#include <network.hpp>

#include <stdint.h>
#include <stdio.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <stdexcept>

void parse_header(FILE* fp, size_t& word_size, std::vector<size_t>& shape) {
  char buffer[256];
  // read useless bytes
  auto ret = fread(buffer, sizeof(char), 11, fp);
  if (ret != 11) throw std::runtime_error("fread failed");
  // get header here
  std::string header = fgets(buffer, 256, fp);

  // find header range
  size_t left, right;
  left = header.find("(");
  right = header.find(")");
  std::regex num_regex("[0-9][0-9]*");
  std::smatch result;
  shape.clear();
  // find shape
  std::string str_shape = header.substr(left + 1, right - left - 1);
  while (std::regex_search(str_shape, result, num_regex)) {
    shape.push_back(std::stoi(result[0].str()));
    str_shape = result.suffix().str();
  }
  // find word_size
  left = header.find("descr");
  left += 11;
  std::string ws = header.substr(left);
  right = ws.find("'");
  word_size = std::stoi(ws.substr(0, right));
}

template <typename T>
void load_npy(FILE* fp, std::vector<T>& out) {
  std::vector<size_t> shape;
  size_t word_size;
  parse_header(fp, word_size, shape);
  out.resize(shape[1]);
  // read data with tpye T into out
  auto ret = fread(out.data(), word_size, shape[1], fp);
  if (ret) return;
}

LAYER_TYPE str_to_type(const std::string& in) {
  if (in == "Conv2d") return CONV;
  if (in == "Linear") return LINEAR;
  if (in == "MaxPool2d") return POOL;
  if (in == "ReLU") return ACT;
  return POOL;
}

template <typename T>
void load_model(Network<T>& net, std::string file_path) {
  std::ifstream in(file_path);
  if (!in) throw std::runtime_error("Open layer.txt error!");
  std::string line, word;
  std::vector<std::string> words;
  // read in lines
  while (std::getline(in, line)) {
    words.clear();
    std::stringstream line_stream(line);
    // read params into word
    while (line_stream >> word) words.push_back(word);
    switch (str_to_type(words[0])) {
      case CONV: {
        // with or without padding
        if (words.size() == 8) {
          auto layer = new Conv2d<T>(std::stoi(words[1]), std::stoi(words[2]),
                                     std::stoi(words[3]), std::stoi(words[4]));
          // load weight
          std::vector<T> weight;
          FILE* fp = fopen(words[6].c_str(), "rb");
          load_npy(fp, weight);
          fclose(fp);
          auto weight_array = xt::adapt(weight, layer->weight_shape());
          layer->set_weight(weight_array);
          // load bias
          std::vector<T> bias;
          fp = fopen(words[7].c_str(), "rb");
          load_npy(fp, bias);
          fclose(fp);
          auto bias_array = xt::adapt(bias, layer->bias_shape());
          layer->set_bias(bias_array);
          // add layer to network
          net << layer;
        } else if (words.size() == 9) {
          auto layer = new Conv2d<T>(std::stoi(words[1]), std::stoi(words[2]),
                                     std::stoi(words[3]), std::stoi(words[4]),
                                     std::stoi(words[5]));
          // load weight
          std::vector<T> weight;
          FILE* fp = fopen(words[7].c_str(), "rb");
          load_npy(fp, weight);
          fclose(fp);
          auto weight_array = xt::adapt(weight, layer->weight_shape());
          layer->set_weight(weight_array);
          // load bias
          std::vector<T> bias;
          fp = fopen(words[8].c_str(), "rb");
          load_npy(fp, bias);
          fclose(fp);
          auto bias_array = xt::adapt(bias, layer->bias_shape());
          layer->set_bias(bias_array);
          // add layer to network
          net << layer;
        } else {
          throw std::runtime_error("Conv2d params error!");
        }
        break;
      }
      case LINEAR: {
        if (words.size() == 6) {
          auto layer = new Linear<T>(std::stoi(words[1]), std::stoi(words[2]));
          // load weight
          std::vector<T> weight;
          FILE* fp = fopen(words[4].c_str(), "rb");
          load_npy(fp, weight);
          fclose(fp);
          auto weight_array = xt::adapt(weight, layer->weight_shape());
          layer->set_weight(weight_array);
          // load bias
          std::vector<T> bias;
          fp = fopen(words[5].c_str(), "rb");
          load_npy(fp, bias);
          fclose(fp);
          auto bias_array = xt::adapt(bias, layer->bias_shape());
          layer->set_bias(bias_array);
          // add layer to network
          net << layer;
        } else {
          throw std::runtime_error("Linear params error!");
        }
        break;
      }
      case ACT: {
        auto layer = new ReLU<T>();
        net << layer;
        break;
      }
      case POOL: {
        if (words.size() == 1) {
          auto layer = new MaxPool2d<T>(2);
          net << layer;
        } else if (words.size() == 4) {
          auto layer = new MaxPool2d<T>(
            std::stoi(words[1]), std::stoi(words[2]), std::stoi(words[3]));
          net << layer;
        } else {
          throw std::runtime_error("MaxPool2d params error!");
        }
        break;
      }
      default: throw std::runtime_error("Unknown layer!");
    }
  }
}

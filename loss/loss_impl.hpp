/**
 * @file loss_impl.hpp
 * @brief loss_impl.hpp
 * @details head file
 * @mainpage mini_nn
 * @author RuiJian Li, YiFan Cao, YanPeng Hu
 * @email lirj@shanghaitech.edu.cn,
 * caoyf@shanghaitech.edu.cn,huyp@shanghaitech.edu.cn
 * @version 1.6.0
 * @date 2019-05-26
 */

template <typename T>
Loss<T>::Loss() {
  /**
   * @brief default use CROSS_ENTROPY
   *
   */
  loss_type_ = CROSS_ENTROPY;
}

template <typename T>
Loss<T>::Loss(LOSS_TYPE loss_type) {
  loss_type_ = loss_type;
}

template <typename T>
T Loss<T>::CrossEntropyLoss(const xt::xarray<T>& scores,
                            const xt::xarray<T>& target) {
  scores_ = scores;
  // scores.shape(): [N, classes]
  // target.shape(): [N]

  /**
   * @brief  /// construct index vector (stupied xt::index_view, maybe bug?)
   *
   */
  xt::xarray<size_t> x = xt::arange(scores.shape(0));
  xt::xarray<size_t> y = xt::cast<size_t>(target);
  auto indexs = x * scores.shape(1) + y;

  auto target_score =
    xt::view(xt::index_view(scores, indexs), xt::all(), xt::newaxis());

  /**
   * @brief exp_sum.shape(): [N, 1]
   *
   */
  /// exp_sum.shape(): [N, 1]
  auto exp_sum =
    xt::view(xt::sum(xt::exp(scores), {1}), xt::all(), xt::newaxis());

  /**
   * @brief loss.shape(): [N, 1]
   *
   */

  auto loss = xt::log(exp_sum) - target_score;
  // TODO: need some optimization here, may cause overflow
  dscores_ = xt::exp(scores) / exp_sum;
  xt::index_view(dscores_, indexs) += -1;
  dscores_ /= scores.shape(0);
  return xt::sum(loss)() / scores.shape(0);
}
template <typename T>
Linear<T>::Linear(size_t in_dims, size_t out_dims) {
  this->layer_type_ = LINEAR;
  Shape W_shape = {in_dims, out_dims};
  Shape b_shape = {out_dims};

  in_dims_ = in_dims;
  out_dims_ = out_dims;

  this->W_ = xt::zeros<T>(W_shape);
  this->b_ = xt::zeros<T>(b_shape);
  this->dW_ = Matrix(W_shape);
  this->db_ = Matrix(b_shape);
  kaiming_normal(*this, "ReLU");
}

template <typename T>
xt::xarray<T> Linear<T>::forward(const xt::xarray<T>& in) {
  this->in_ = in;
  in_reshape_ = Matrix(in);
  in_reshape_.reshape({in.shape(0), in_dims_});
  Matrix out = xt::linalg::dot(in_reshape_, this->W_) + this->b_;
  return out;
}

template <typename T>
xt::xarray<T> Linear<T>::backward(const xt::xarray<T>& dout) {
  this->db_ = xt::sum(dout, {0});
  this->dW_ = xt::linalg::dot(xt::transpose(in_reshape_), dout);
  this->din_ = xt::linalg::dot(dout, xt::transpose(this->W_));
  this->din_.reshape(this->in_.shape());

  this->net_->get_optimizer()->update(this->W_, this->dW_);
  this->net_->get_optimizer()->update(this->b_, this->db_);

  return this->din_;
}

template <typename T>
size_t Linear<T>::get_fan() {
  return this->W_.shape(1);
}
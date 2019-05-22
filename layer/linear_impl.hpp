template <typename T>
Linear<T>::Linear(size_t in_dims, size_t out_dims) {
  Shape W_shape = {in_dims, out_dims};
  Shape b_shape = {out_dims};

  this->in_dims_ = in_dims;
  this->out_dims_ = out_dims;

  this->W_ = xt::zeros<T>(W_shape);
  this->b_ = xt::zeros<T>(b_shape);
  this->dW_ = Matrix(W_shape);
  this->db_ = Matrix(b_shape);
}

template <typename T>
xt::xarray<T> Linear<T>::forward(const xt::xarray<T>& in) {
  if (in.shape().size() != 4) {
    std::cerr << "Input shape: (N, C, H, W) " << std::endl;
    exit(0);
  }
  this->in_ = in;
  this->in_reshape_ = Matrix(in);
  this->in_reshape_.reshape({in.shape(0), this->in_dims_});
  Matrix out = xt::linalg::dot(this->in_reshape_, this->W_) + this->b_;
  return out;
}

template <typename T>
xt::xarray<T> Linear<T>::backward(const xt::xarray<T>& dout) {
  this->db_ = xt::sum(dout, {0});
  this->dW_ = xt::linalg::dot(xt::transpose(this->in_reshape_), dout);

  this->din_ = xt::linalg::dot(dout, xt::transpose(this->W_));
  this->din_.reshape(this->in_.shape());

  return this->din_;
}
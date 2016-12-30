# Leaf Classification

Repo for [Kaggle Leaf Classification challenge](https://www.kaggle.com/c/leaf-classification/).

Code currently is just rough experimental stuff, and does not follow best practices.

The network architecture is a small convnet, using `3x3` and `1x1` convolutional kernels. Global average pooling occurs instead of a fully connected layer for aggregration, this allows any input volume size to be used for training and inference. Optionally the given features from `Kaggle` may be used, and concatenated with the vector of extracted CNN codes, and aggregated using a `Fully Connected` layer.

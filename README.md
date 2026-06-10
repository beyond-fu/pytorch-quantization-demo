# pytorch-quantization-demo

A simple network quantization demo using pytorch from scratch. This is the code for my [tutorial](https://mp.weixin.qq.com/s?__biz=Mzg4ODA3MDkyMA==&mid=2247483692&idx=1&sn=3e28db4881d591f4e6a66c83d4213823&chksm=cf81f74bf8f67e5d0f2a98fd7bf7a91864d14010d88a5ed89120b7b4fcd94fc34789f0d0db9a&token=680347690&lang=zh_CN#rd) about network quantization written in Chinese.

# TRAIN

- create model instance
- create optimizer used for train
- iter for epoch
  - train
    - switch to train mode
    - create loss function
    - iter for all train batches(train sets)
      - clear grad of optimizer
      - calling `forward` function of model
      - get loss and backpropogation
      - update weights of model using optimizer, momentum and calculated grad of last step
  - test
    - switch to eval mode
    - create loss function
    - iter for all test batches(test sets)
      - calling `forward` function of model
      - accumulate loss
      - accumulate num of correct
    - calc total_loss and accuracy
- model(data) has the same calling order no matter model.eval() or model.train()

# INFERENCE

-

# naive_cnn
A simple convolutional neural network implementation created to learn how CNNs work under the hood. Uses Cupy to do almost all of the calculations which means you'll need an Nvidia GPU to run it. 

The network architecture is simple, one convolution layer with 6 sets of 3 filters of size 3x3, one max pooling layer with 2x2 blocks and stride of 2 after which it has a ReLU activation layer. This feeds into a dense layer of 4050 with ReLU which finally feeds into a dense layer of 10 with softmax. The error function used is Cross-Entropy loss. The network has an accuracy of ~54% after training for 10 epochs. 

### Problems
* It works very slowly, taking more than 2 hours to train on CIFAR-10 on an RTX 2070 Super. I suspect there are two problems. The first is that the convolution (or more correctly the correlation) function is just a naive one and can be rewritten to use matrix multiplication which is a much faster operation on the GPU. The second be that some data is being sent back and forth from RAM to VRAM during every iteration of the main loop. I'd have to use the CUDA profiler to investigate further. A final speed up could be achieved using Numba to compile.
* The code isn't elegant at all and will probably be rewritten to be general and make it easier to stack layers without having to rewrite the glue. At the moment, it is just a set of functions.
* It doesn't use any of the newer weight initialization techniques such as Glorot or He initialization.

### Running the code
You'll need the CIFAR-10 data split into 50000 and 10000 samples in files called train_data.mat and test_data.mat to run it. Simply run ```python train_cnn.py``` to train and test it.

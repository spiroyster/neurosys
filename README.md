# Neurosys

C++ header only Nueral network framework. C++11 & STL only.

## Usage

Simply include neurosys.hpp and you're off.

    #include "neurosys.hpp"

The neurosys namespace contains everything required to create neural networks and start training...

### Current support.

* feedforward neural networks.
* backpropagation through gd and sgd.
* Sigmoid, softmax and tanh activations.
* Squared Error  (MSE) and Cross Entropy (CE) loss functions.

# Build

neurosys is a header only library so no compilation for the library is required. There are examples and tests which are included and these are built and deposited in /neurosys/bin.

## win (VS2017)

VS2017 solution file can be found /msvc/neurosys.sln and contains all the test and example projects.

## *nix (GCC)

Makefile supplied for bulding with gcc.

To build the tests...

    make build_test

To build the examples...

    make build_examples

Or spcifically build the various examples...

    make build_example_xor


## Roadmap

Ideally the end goal is to support CNN, RNN and various other generative machine learning algorithms to perform style transfer and hopefully somthing akin to StyleGAN.

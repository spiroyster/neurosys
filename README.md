# About

C++ header only Nueral network framework. C++11 & STL only.

## Usage

Simply include neurosys.hpp and you're off.

    #include "neurosys.hpp"

The neurosys namespace contains everything required to create neural networks and train these networks to learn and predict.

Current support:

* feedforward neural networks.
* backpropagation through gd and sgd.

## Build

## nix

There is a makefile at root which will build binaries and place them in bin folder.

To build the tests...

    make build_test

To build the examples...

    make build_examples

Or spcifically build the various examples...

    make build_example_xor

### win

VS2017 solution file can be found /msvc/neurosys.sln and contains all the test and example projects which will be built into bin.

# Roadmap

Ideally the end goal is to support CNN, RNN and various other generative machine learning algorithms to perform style transfer and hopefully somthing akin to StyleGAN.

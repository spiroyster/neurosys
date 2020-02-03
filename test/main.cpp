#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "..\include\neurosys.hpp"

namespace
{
	bool ApproxEquals(const double a, const double b, unsigned int dp)
	{
		double numerDenom = static_cast<double>(pow(10, dp));
		return std::round(a * numerDenom) / numerDenom == std::round( b *numerDenom) / numerDenom;
	}
}

TEST_CASE("linear activation", "[activation]")
{
	CHECK(neurosys::activation::linear(-1.0) == -1.0);
	CHECK(neurosys::activation::linear(-0.1) == -0.1);
	CHECK(neurosys::activation::linear(-0.01) == -0.01);
	CHECK(neurosys::activation::linear(-0.00001) == -0.00001);
	CHECK(neurosys::activation::linear(0) == 0);
	CHECK(neurosys::activation::linear(1.0) == 1.0);
	CHECK(neurosys::activation::linear(0.1) == 0.1);
	CHECK(neurosys::activation::linear(0.01) == 0.01);
	CHECK(neurosys::activation::linear(0.00001) == 0.00001);
}

TEST_CASE("ReLU activation", "[activation]")
{
	CHECK(neurosys::activation::ReLU(-1.0) == 0);
	CHECK(neurosys::activation::ReLU(-0.1) == 0);
	CHECK(neurosys::activation::ReLU(-0.01) == 0);
	CHECK(neurosys::activation::ReLU(-0.00001) == 0);
	CHECK(neurosys::activation::ReLU(0) == 0);
	CHECK(neurosys::activation::ReLU(1.0) == 1.0);
	CHECK(neurosys::activation::ReLU(0.1) == 0.1);
	CHECK(neurosys::activation::ReLU(0.01) == 0.01);
	CHECK(neurosys::activation::ReLU(0.00001) == 0.00001);
}

TEST_CASE("leakyReLU activation", "[activation]")
{
	/*CHECK(neurosys::activation::leakyReLU(-1.0) == 0);
	CHECK(neurosys::activation::leakyReLU(-0.1) == 0);
	CHECK(neurosys::activation::leakyReLU(-0.01) == 0);
	CHECK(neurosys::activation::leakyReLU(-0.00001) == 0);
	CHECK(neurosys::activation::leakyReLU(0) == 0);
	CHECK(neurosys::activation::leakyReLU(1.0) == 1.0);
	CHECK(neurosys::activation::leakyReLU(0.1) == 0.1);
	CHECK(neurosys::activation::leakyReLU(0.01) == 0.01);
	CHECK(neurosys::activation::leakyReLU(0.00001) == 0.00001);*/
}

TEST_CASE("sigmoid activation", "[activation]")
{
	/*CHECK(neurosys::activation::leakyReLU(-1.0) == 0);
	CHECK(neurosys::activation::leakyReLU(-0.1) == 0);
	CHECK(neurosys::activation::leakyReLU(-0.01) == 0);
	CHECK(neurosys::activation::leakyReLU(-0.00001) == 0);
	CHECK(neurosys::activation::leakyReLU(0) == 0);
	CHECK(neurosys::activation::leakyReLU(1.0) == 1.0);
	CHECK(neurosys::activation::leakyReLU(0.1) == 0.1);
	CHECK(neurosys::activation::leakyReLU(0.01) == 0.01);
	CHECK(neurosys::activation::leakyReLU(0.00001) == 0.00001);*/
}

// layer construction...
TEST_CASE("layer construction", "[layer]")
{
	{
		neurosys::layer l({ 0.1, 0.2, 0.3 }, neurosys::activation::linear);

		CHECK(l.neurons_.size() == 3);
		CHECK(l.neurons_[0] == 0.1);
		CHECK(l.neurons_[1] == 0.2);
		CHECK(l.neurons_[2] == 0.3);
		CHECK(l.activation_(1.0) == 1.0);
		CHECK(l.bias_ == 0);
	}
	
	{
		neurosys::layer l({ 0.1, 0.2, 0.3 }, neurosys::activation::linear, 0.4);

		CHECK(l.neurons_.size() == 3);
		CHECK(l.neurons_[0] == 0.1);
		CHECK(l.neurons_[1] == 0.2);
		CHECK(l.neurons_[2] == 0.3);
		CHECK(l.activation_(1.0) == 1.0);
		CHECK(l.bias_ == 0.4);
	}

	{
		neurosys::layer l(3, neurosys::activation::linear);

		CHECK(l.neurons_.size() == 3);
		CHECK(l.activation_(1.0) == 1.0);
		CHECK(l.bias_ == 0);
	}

	{
		neurosys::layer l(3, neurosys::activation::linear, 0.4);

		CHECK(l.neurons_.size() == 3);
		CHECK(l.activation_(1.0) == 1.0);
		CHECK(l.bias_ == 0.4);
	}
	
}

// layer largest 
TEST_CASE("layer largest", "[layer]")
{
	{
		neurosys::layer l({ 0.1, 0.2, 0.3, 0.4, 0.5 }, neurosys::activation::linear);
		CHECK(l.largest() == 4);
	}
	{
		neurosys::layer l({ 0.5, 0.4, 0.3, 0.2, 0.1 }, neurosys::activation::linear);
		CHECK(l.largest() == 0);
	}
	{
		neurosys::layer l({ 0.1, 0.1, 0.2, 0.1, 0.1 }, neurosys::activation::linear);
		CHECK(l.largest() == 2);
	}
}

// layer sum
TEST_CASE("layer sum", "[layer]")
{
	{
		neurosys::layer l({ 0.1, 0.2, 0.3, 0.4, 0.5 }, neurosys::activation::linear);
		CHECK(l.sum() == 1.5);
	}
	{
		neurosys::layer l({ 0.5, 0.4, 0.3, 0.2, 0.1 }, neurosys::activation::linear);
		CHECK(l.sum() == 1.5);
	}
	{
		neurosys::layer l({ 0.1, 0.1, 0.2, 0.1, 0.1 }, neurosys::activation::linear);
		CHECK(l.sum() == 0.6);
	}
}

// layer squared error
TEST_CASE("layer squaredError", "[layer]")
{
	{
		neurosys::layer i({ 0.1, 0.2, 0.3, 0.4, 0.5 }, neurosys::activation::linear);
		neurosys::layer j({ 0.6, 0.7, 0.8, 0.9, 1.0 }, neurosys::activation::linear);
		CHECK(i.squaredError(j) == 0.25);
	}

}

// layer assignment
TEST_CASE("layer assignment", "[layer]")
{
	neurosys::layer l({ 0.1, 0.2, 0.3, 0.4, 0.5 }, neurosys::activation::linear);
	neurosys::layer ll = l;

	CHECK(l == ll);
}


// network construction
TEST_CASE("network construction", "[network]")
{
	{
		neurosys::network net({
		neurosys::layer(1, neurosys::activation::linear),
		neurosys::layer(1, neurosys::activation::linear)
			});

		CHECK(net.layers_.size() == 2);
		CHECK(net.layers_[0].neurons_.size() == 1);
		CHECK(net.layers_[1].neurons_.size() == 1);
		CHECK(net.weights_.size() == net.layers_.size() - 1);
		CHECK(net.weights_[0].size() == 1);
	}

	{
		neurosys::network net({
		neurosys::layer(2, neurosys::activation::linear),
		neurosys::layer(2, neurosys::activation::linear)
			});

		CHECK(net.layers_.size() == 2);
		CHECK(net.layers_[0].neurons_.size() == 2);
		CHECK(net.layers_[1].neurons_.size() == 2);
		CHECK(net.weights_.size() == net.layers_.size() - 1);
		CHECK(net.weights_[0].size() == net.layers_[0].neurons_.size() * net.layers_[1].neurons_.size());
	}

	{
		neurosys::network net({
		neurosys::layer(3, neurosys::activation::linear),
		neurosys::layer(3, neurosys::activation::linear)
			});

		CHECK(net.layers_.size() == 2);
		CHECK(net.layers_[0].neurons_.size() == 3);
		CHECK(net.layers_[1].neurons_.size() == 3);
		CHECK(net.weights_.size() == net.layers_.size() - 1);
		CHECK(net.weights_[0].size() == net.layers_[0].neurons_.size() * net.layers_[1].neurons_.size());
	}

	{
		neurosys::network net({
		neurosys::layer(1, neurosys::activation::linear),
		neurosys::layer(2, neurosys::activation::linear),
		neurosys::layer(3, neurosys::activation::linear),
		neurosys::layer(4, neurosys::activation::linear)
			});

		CHECK(net.layers_.size() == 4);
		CHECK(net.layers_[0].neurons_.size() == 1);
		CHECK(net.layers_[1].neurons_.size() == 2);
		CHECK(net.layers_[2].neurons_.size() == 3);
		CHECK(net.layers_[3].neurons_.size() == 4);
		CHECK(net.weights_.size() == net.layers_.size() - 1);
		
		CHECK(net.weights_[0].size() == net.layers_[0].neurons_.size() * net.layers_[1].neurons_.size());
		CHECK(net.weights_[1].size() == net.layers_[1].neurons_.size() * net.layers_[2].neurons_.size());
		CHECK(net.weights_[2].size() == net.layers_[2].neurons_.size() * net.layers_[3].neurons_.size());
	}
	
}

// network reset...


TEST_CASE("network weight", "[network]")
{
	neurosys::network net({
		neurosys::layer({ 0.0, 1.0, 2.0 }, neurosys::activation::linear),
		neurosys::layer({ 3.0, 4.0 }, neurosys::activation::linear),
		neurosys::layer({ 5.0, 6.0 }, neurosys::activation::sigmoid, 0.5)
		});

	net.weight(0, 0, 0) = 0.1;
	CHECK(net.weights_[0][0] == 0.1);
	
	// ...
}


// network weighting...
TEST_CASE("network weights", "[network]")
{
	neurosys::network net({
		neurosys::layer({ 0.0, 1.0, 2.0 }, neurosys::activation::linear),
		neurosys::layer({ 3.0, 4.0 }, neurosys::activation::linear),
		neurosys::layer({ 5.0, 6.0 }, neurosys::activation::sigmoid, 0.5)
		});

	net.weights(0, 0, { 0.1, 0.2 });
	CHECK(net.weight(0, 0, 0) == 0.1);
	CHECK(net.weight(0, 0, 1) == 0.2);

	net.weights(0, 1, { 1.1, 1.2 });
	CHECK(net.weight(0, 1, 0) == 1.1);
	CHECK(net.weight(0, 1, 1) == 1.2);

	net.weights(0, 2, { 2.1, 2.2 });
	CHECK(net.weight(0, 2, 0) == 2.1);
	CHECK(net.weight(0, 2, 1) == 2.2);

	net.weights(1, 0, { 3.1, 3.2 });
	CHECK(net.weight(1, 0, 0) == 3.1);
	CHECK(net.weight(1, 0, 1) == 3.2);

	net.weights(1, 1, { 4.1, 4.2 });
	CHECK(net.weight(1, 1, 0) == 4.1);
	CHECK(net.weight(1, 1, 1) == 4.2);

}

TEST_CASE("network forwardWeights", "[network]")
{
	neurosys::network net({
		neurosys::layer({ 0.0, 1.0, 2.0 }, neurosys::activation::linear),
		neurosys::layer({ 3.0, 4.0 }, neurosys::activation::linear),
		neurosys::layer({ 5.0, 6.0 }, neurosys::activation::sigmoid, 0.5)
		});

	net.weights(0, 0, { 0.1, 0.2 });
	net.weights(0, 1, { 1.1, 1.2 });
	net.weights(0, 2, { 2.1, 2.2 });
	net.weights(1, 0, { 3.1, 3.2 });
	net.weights(1, 1, { 4.1, 4.2 });

	{
		std::vector<neurosys::neuron*> x1 = net.forwardWeights(0, 0);
		CHECK(x1.size() == 2);
		CHECK(*x1[0] == 0.1);
		CHECK(*x1[1] == 0.2);
	}

	{
		std::vector<neurosys::neuron*> x2 = net.forwardWeights(0, 1);
		CHECK(x2.size() == 2);
		CHECK(*x2[0] == 1.1);
		CHECK(*x2[1] == 1.2);
	}

	{
		std::vector<neurosys::neuron*> x3 = net.forwardWeights(0, 2);
		CHECK(x3.size() == 2);
		CHECK(*x3[0] == 2.1);
		CHECK(*x3[1] == 2.2);
	}

	{
		std::vector<neurosys::neuron*> h1 = net.forwardWeights(1, 0);
		CHECK(h1.size() == 2);
		CHECK(*h1[0] == 3.1);
		CHECK(*h1[1] == 3.2);
	}

	{
		std::vector<neurosys::neuron*> h2 = net.forwardWeights(1, 1);
		CHECK(h2.size() == 2);
		CHECK(*h2[0] == 4.1);
		CHECK(*h2[1] == 4.2);
	}
		
}

// backward weight
TEST_CASE("network backwardWeight", "[network]")
{
	neurosys::network net({
		neurosys::layer({ 0.0, 1.0, 2.0 }, neurosys::activation::linear),
		neurosys::layer({ 3.0, 4.0 }, neurosys::activation::linear),
		neurosys::layer({ 5.0, 6.0 }, neurosys::activation::sigmoid, 0.5)
		});

	net.weights(0, 0, { 0.1, 0.2 });
	net.weights(0, 1, { 1.1, 1.2 });
	net.weights(0, 2, { 2.1, 2.2 });
	net.weights(1, 0, { 3.1, 3.2 });
	net.weights(1, 1, { 4.1, 4.2 });

	{
		std::vector<neurosys::neuron*> o1 = net.backWeights(2, 0);
		CHECK(o1.size() == 2);
		CHECK(*o1[0] == 3.1);
		CHECK(*o1[1] == 4.1);
	}

	{
		std::vector<neurosys::neuron*> o2 = net.backWeights(2, 1);
		CHECK(o2.size() == 2);
		CHECK(*o2[0] == 3.2);
		CHECK(*o2[1] == 4.2);
	}

	{
		std::vector<neurosys::neuron*> h1 = net.backWeights(1, 0);
		CHECK(h1.size() == 3);
		CHECK(*h1[0] == 0.1);
		CHECK(*h1[1] == 1.1);
		CHECK(*h1[2] == 2.1);
	}

	{
		std::vector<neurosys::neuron*> h2 = net.backWeights(1, 1);
		CHECK(h2.size() == 3);
		CHECK(*h2[0] == 0.2);
		CHECK(*h2[1] == 1.2);
		CHECK(*h2[2] == 2.2);
	}

}

// dot
TEST_CASE("feedForward dot", "[feedForward]")
{
	CHECK(neurosys::feedForward::dot({ 0, 1.0 }, 0, { 0, 1.0 }) == 1.0);
	CHECK(neurosys::feedForward::dot({ 1.0, 2.0 }, 0, { 3.0, 4.0 }) == 11.0);
	CHECK(neurosys::feedForward::dot({ 1.0, 2.0, 3.0 }, 0, { 4.0, 5.0, 6.0 }) == 32.0);
	CHECK(neurosys::feedForward::dot({ -1.0, 2.0 }, 0, { 3.0, -4.0 }) == -11.0);
	CHECK(neurosys::feedForward::dot({ -1.0, 2.0, 3.0 }, 0, { 4.0, 5.0, 6.0 }) == 24.0);
}

// multiply
TEST_CASE("feedForward feed", "[feedForward network]")
{
	// Create the network with empty activation values and 0.5 bias...
	neurosys::network net({
		neurosys::layer({ 1.0, 2.0, 3.0 }, neurosys::activation::linear),
		neurosys::layer({ 4.0, 5.0 }, neurosys::activation::sigmoid, 0.5),
		neurosys::layer({ 6.0, 7.0 }, neurosys::activation::sigmoid, 0.5)
		});

	// we need to set the weights...
	net.weights(0, 0, { 0.1, 0.2 });
	net.weights(0, 1, { 0.3, 0.4 });
	net.weights(0, 2, { 0.5, 0.6 });
	net.weights(1, 0, { 0.7, 0.8 });
	net.weights(1, 1, { 0.9, 0.1 });

	// single feed forward and check hidden layer values...
	neurosys::layer result = neurosys::feedForward::feed(net.layers_[0], net.layers_[1], net.weights_[0]);

	// check the result is the same as the hidden layer...
	CHECK(result.neurons_.size() == net.layers_[1].neurons_.size());
	for (unsigned int i = 0; i < result.neurons_.size(); ++i) 
	{
		Approx target = Approx(result.neurons_[i]).epsilon(0.01);
		CHECK(target == net.layers_[1].neurons_[i]);
	}
		
	result = neurosys::feedForward::feed(net.layers_[1], net.layers_[2], net.weights_[1]);

	// check the result is the same as the expected output layer...
	CHECK(result.neurons_.size() == net.layers_[2].neurons_.size());
	for (unsigned int i = 0; i < result.neurons_.size(); ++i)
	{
		Approx target = Approx(result.neurons_[i]).epsilon(0.01);
		CHECK(target == net.layers_[2].neurons_[i]);
	}
}




// back propagation...


TEST_CASE("feedForward observation", "[feedForward network]")
{
	// https://www.anotsorandomwalk.com/backpropagation-example-with-numbers-step-by-step/
	
	// Create the network with empty activation values and 0.5 bias...
	neurosys::network net({
		neurosys::layer({ 1.0, 4.0, 5.0 }, neurosys::activation::linear),
		neurosys::layer({ 0.9866, 0.9950 }, neurosys::activation::sigmoid, 0.5),
		neurosys::layer({ 0.8896, 0.8004 }, neurosys::activation::sigmoid, 0.5)
		});

	// we need to set the weights...
	net.weights(0, 0, { 0.1, 0.2 });
	net.weights(0, 1, { 0.3, 0.4 });
	net.weights(0, 2, { 0.5, 0.6 });
	net.weights(1, 0, { 0.7, 0.8 });
	net.weights(1, 1, { 0.9, 0.1 });

	// single feed forward and check hidden layer values...
	neurosys::layer result = neurosys::feedForward::feed(net.layers_[0], net.layers_[1], net.weights_[0]);

	// check the result is the same as the hidden layer...
	CHECK(result.neurons_.size() == net.layers_[1].neurons_.size());
	for (unsigned int i = 0; i < result.neurons_.size(); ++i)
	{
		Approx target = Approx(result.neurons_[i]).epsilon(0.01);
		CHECK(target == net.layers_[1].neurons_[i]);
	}

	result = neurosys::feedForward::feed(net.layers_[1], net.layers_[2], net.weights_[1]);

	// check the result is the same as the expected output layer...
	CHECK(result.neurons_.size() == net.layers_[2].neurons_.size());
	for (unsigned int i = 0; i < result.neurons_.size(); ++i)
	{
		Approx target = Approx(result.neurons_[i]).epsilon(0.01);
		CHECK(target == net.layers_[2].neurons_[i]);
	}

	// t1 = 0.1, t2 = 0.05

	// Check the squares error...
	//neurosys::layer expected

	// Perform the back propagation....



}


// https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

	// https://cs.stanford.edu/people/karpathy/convnetjs/intro.html


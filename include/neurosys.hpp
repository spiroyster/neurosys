#ifndef NEUROSYS_HPP
#define NEUROSYS_HPP

#include <fstream>
#include <vector>
#include <list>
#include <functional>
#include <random>
#include <numeric>
#include <assert.h>

namespace neurosys
{
	typedef double neuron;

	namespace activation
	{
		typedef std::function<neuron(const neuron& x)> fn;

		static fn linear = [](const neuron& x) { return x; };
		static fn ReLU = [](const neuron& x) { return x > 0 ? x : 0; }; 
		static fn leakyReLU = [](const neuron& x) { return x > 0 ? x : 0.01 * x; };
		static fn fastSigmoid = [](const neuron& x) { return x / (1.0+abs(x)); };
		static fn sigmoid= [](const neuron& x) { return 1.0 / ( 1.0 + exp(-x)); };
	}

	struct layer
	{
	public:
		layer(const std::vector<neuron>& neurons, activation::fn activation) : neurons_(neurons), activation_(activation), bias_(0) {}
		layer(const std::vector<neuron>& neurons, activation::fn activation, double bias) : neurons_(neurons), activation_(activation), bias_(bias) {}
		layer(unsigned int size, activation::fn activation) : neurons_(size, 0), activation_(activation), bias_(0) {}
		layer(unsigned int size, activation::fn activation, double bias) : neurons_(size, 0), activation_(activation), bias_(bias) {}
		
		layer& operator=(const layer& rhs)
		{
			neurons_ = rhs.neurons_;
			activation_ = rhs.activation_;
			bias_ = rhs.bias_;
			return *this;
		}

		bool operator==(const layer& rhs) const
		{
			return neurons_ == rhs.neurons_ && bias_ == rhs.bias_; // also check activation functions are the same?
		}
		bool operator!=(const layer& rhs) const
		{
			return !(*this == rhs);
		}

		std::size_t largest()
		{
			return std::distance(neurons_.begin(), std::max_element(neurons_.begin(), neurons_.end()));
		}

		neuron sum()
		{
			return std::accumulate(neurons_.begin(), neurons_.end(), 0.0);
		}

		layer squaredError(const layer& expected) const
		{
			// assert layers have same number of items...
			assert(neurons_.size() == expected.neurons_.size());

			layer result = expected;
			for (unsigned int n = 0; n < neurons_.size(); ++n)
				result.neurons_[n] = (result.neurons_[n] - neurons_[n]) * (result.neurons_[n] - neurons_[n]);
			return result;
		}

		double sumSquaredError(const layer& expected) const
		{
			layer result = squaredError(expected);
			return result.sum() / static_cast<double>(neurons_.size());
		}

		std::vector<neuron> neurons_;
		activation::fn activation_;
		double bias_;
	};

	struct network
	{
		network(const std::vector<layer>& layers) : layers_(layers)
		{
			assert(layers_.size() >= 2);

			// construct the weight matrices...
			weights_.reserve(layers_.size() - 1);
			for (unsigned int l = 0; l < layers_.size() - 1; ++l)
				weights_.push_back(std::vector<neuron>(layers_[l].neurons_.size() * layers_[l+1].neurons_.size()));
		}

		void reset()
		{
			std::default_random_engine rd;
			std::mt19937 eng(rd());
			std::uniform_real_distribution<double> dist(0.0, 1.0);
			for (unsigned int l = 0; l < layers_.size(); ++l)
			{
				for (unsigned int n = 0; n < layers_[l].neurons_.size(); ++n)
					layers_[l].neurons_[n] = dist(eng);
				layers_[l].bias_ = dist(eng);
			}

			for (unsigned int w = 0; w < weights_.size(); ++w)
				for (unsigned int n = 0; n < weights_[w].size(); ++n)
					weights_[w][n] = dist(eng);
		}

		// l1 = layer, l1n = layer1 neuron, l2n = layer2 neuron
		neuron& weight(unsigned int l1, unsigned int l1n, unsigned int l2n)
		{
			assert(l1 < layers_.size() - 1);
			assert(l1n < layers_[l1].neurons_.size());
			assert(l2n < layers_[l1 + 1].neurons_.size());

			return weights_[l1][(layers_[l1].neurons_.size() * l2n) + l1n];
		}

		// l1 = layer, l1n = layer1 neuron, l2n = layer2 neuron
		const neuron& weight(unsigned int l1, unsigned int l1n, unsigned int l2n) const
		{
			assert(l1 < layers_.size() - 1);
			assert(l1n < layers_[l1].neurons_.size());
			assert(l2n < layers_[l1 + 1].neurons_.size());

			return weights_[l1][(layers_[l1].neurons_.size() * l2n) + l1n];
		}

		// l1 = layer, l1n = layer1 neuron, l1weights = weights for l1 neuron
		void weights(unsigned int l1, unsigned int l1n, const std::vector<neuron>& l1nweights)
		{
			assert(l1 < layers_.size() - 1);
			assert(l1n < layers_[l1].neurons_.size());
			assert(l1nweights.size() == layers_[l1 + 1].neurons_.size());

			std::size_t l1size = layers_[l1].neurons_.size();
			
			for (std::size_t n = 0; n < l1nweights.size(); ++n)
				weights_[l1][(n * l1size) + l1n] = l1nweights[n];
		}

		// given a neuron, what are its weights to all neurons of the next layer...
		std::vector<neuron*> forwardWeights(unsigned int l1, unsigned int l1n)
		{
			// assert not last layer...
			assert(l1 != layers_.size());
			assert(l1n < layers_[l1].neurons_.size());
			
			std::size_t l1size = layers_[l1].neurons_.size();
			std::size_t l2size = layers_[l1 + 1].neurons_.size();

			std::vector<neuron*> result(l2size);
			for (std::size_t n = 0; n < l2size; ++n)
				result[n] = &weights_[l1][(l1size * n) + l1n] ;
			return result;
		}

		// given a neuron, what are all the weights from the previous layer to the neuron...
		std::vector<neuron*> backWeights(unsigned int l2, unsigned int l2n)
		{
			// assert not first layer...
			assert(l2 > 0);
			assert(l2n < layers_[l2-1].neurons_.size());

			std::size_t l1size = layers_[l2 - 1].neurons_.size();
			std::vector<neuron*> result(l1size);
			
			for (std::size_t n = 0; n < l1size; ++n)
				result[n] = &weights_[l2 - 1][(l1size * l2n) + n];
			return result;
		}

		std::vector<layer> layers_;
		std::vector<std::vector<neuron>> weights_;
	};


	namespace feedForward
	{
		neuron dot(const std::vector<neuron>& a, unsigned int aStart, const std::vector<neuron>& b)
		{
			neuron result = 0;
			for (unsigned int i = aStart, j = 0; j < b.size(); ++i, ++j)
				result += a[i] * b[j];
			return result;
		}

		layer feed(const layer& currentLayer, const layer& nextLayer, const std::vector<neuron>& weights)
		{
			layer result = nextLayer;
			for (unsigned int r = 0; r < result.neurons_.size(); ++r)
				result.neurons_[r] = result.activation_(dot(weights, r * static_cast<unsigned int>(currentLayer.neurons_.size()), currentLayer.neurons_) + nextLayer.bias_);
			return result;
		}

		layer observation(const network& n, const layer& input)
		{
			layer result = input;
			for (unsigned int hl = 1; hl < n.layers_.size(); ++hl)
				result = feed(result, n.layers_[hl], n.weights_[hl-1]);
			return result;
		}
	}
	

	// calculate error...

	// calculate cost...

	// backPropagation
	namespace backPropagation
	{
		// back propagation of weights...
		std::vector<neuron> backPropagate(const network& n, unsigned int l2, unsigned int l1, const layer& result, const layer& expected)
		{

		}

		std::vector<neuron> backPropagate(const network& n, unsigned int l2, unsigned int l1, const layer& result, const layer& expected)
		{

		}


		// for each layer going backwards...
		/*layer propagate(const network& n, unsigned int l2, unsigned int l1, const layer& result, const layer& expected)
		{



			return layer();
		}*/



	}


}

#endif // NEUROSYS_HPP
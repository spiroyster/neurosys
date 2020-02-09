#ifndef NEUROSYS_HPP
#define NEUROSYS_HPP

#include <vector>
#include <functional>
#include <assert.h>
#include <random>


namespace neurosys
{
	namespace activation
	{
		typedef std::function<double(const double& x)> activationFn;

		enum activation
		{
			sigmoid,
			linear
		};

		std::vector<activationFn> Fn
		{
			[](const double& x) { return 1.0 / (1.0 + exp(-x)); },
			[](const double& x) { return x; }
		};

		std::vector<activationFn> FnPrime
		{
			[](const double& x) { return (1.0 - x); },
			[](const double& x) { return 1.0; }
		};
	}


	class matrix
	{
	public:
		matrix(unsigned int m, unsigned int n) : values_(m * n, 0), stride_(n) {}
		matrix(const std::vector<double>& values, unsigned int stride) : values_(values), stride_(stride) {}
		
		const double& operator[](unsigned int i) const { return values_[i]; }
		double& operator[](unsigned int i) { return values_[i]; }
		
		const double& value(unsigned int m, unsigned int n) const { return values_[(m * stride_) + n]; }
		double& value(unsigned int m, unsigned int n) { return values_[(m * stride_) + n]; }
		const std::vector<double>& values() const { return values_; }
		std::vector<double>& values() { return values_; }

		unsigned int m() const { return stride_ ? static_cast<unsigned int>(values_.size() / stride_) : 0; }
		unsigned int n() const { return stride_; }
		unsigned int size() const { return static_cast<unsigned int>(values_.size()); }

		bool isColumnVector() const { return n() == 1 && m() != 0; }
		bool isRowVector() const { return m() == 1 && n() != 0; }

	protected:
		std::vector<double> values_;
		unsigned int stride_;		// number of elements in a row.
	};
	
	class neurons : public matrix
	{
	public:
		neurons(unsigned int n) : matrix(n, 1) {}
		neurons(const std::vector<double>& values) : matrix(values, 1) {}

		const double& value(unsigned int n) const { return values_[n]; }
		double& value(unsigned int n) { return values_[n]; }

	};


	// layer (weight matrix), bias...
	class layer
	{
	public:
		layer(unsigned int neuronCount, const activation::activation& a, double bias)
			: weights_(neuronCount, 1), bias_(bias), activation_(a)
		{
		}
		layer(unsigned int neuronCount, const activation::activation& a)
			: weights_(neuronCount, 1), bias_(1.0), activation_(a)
		{
		}
				
		layer(const matrix& weights, const activation::activation& a, double bias) 
			: weights_(weights), bias_(bias), activation_(a) 
		{
		}

		const matrix& weights() const { return weights_; }
		matrix& weights() { return weights_; }

		double bias() const { return bias_; }
		const activation::activation& activation() const { return activation_; }
		
		void bias(double b) { bias_ = b; }
		void activation(const activation::activation& a) { activation_ = a; }

	protected:
		matrix weights_;
		double bias_;
		activation::activation activation_;
	};

	class input : public layer
	{
	public:
		input(unsigned int neuronCount)
			:	layer(neuronCount, activation::linear, 1.0)
		{
		}

		input(const std::vector<double>& values)
			: layer(matrix(values, 1), activation::linear, 1.0)
		{
		}

		double& neuron(unsigned int n) { return weights_[n]; }
	};

	class output : public layer
	{
	public:
		output(unsigned int neuronCount, const activation::activation& a, double bias)
			: layer(neuronCount, a, bias)
		{
		}
	};

	
	// network
		// multiple layers..

	class network
	{
	public:
		network(const input& i, const std::vector<layer>& hidden, const output& o)
		{
			if (hidden.empty())
				layers_ = { layer(matrix(i.weights().size(), o.weights().size()), i.activation(), i.bias()), o };
			else
			{
				layers_.reserve(hidden.size() + 2);
				layers_.push_back(layer(matrix(i.weights().size(), hidden.front().weights().size()), hidden.front().activation(), hidden.front().bias()));
				for (unsigned int h = 0; h < (hidden.size() - 1); ++h)
					layers_.push_back(layer(matrix(hidden[h].weights().size(), hidden[h + 1].weights().size()), hidden[h].activation(), hidden[h].bias()));
				layers_.push_back(layer(matrix(hidden.back().weights().size(), o.weights().size()), hidden.back().activation(), hidden.back().bias()));
				layers_.push_back(o);
			}
		}

		const layer& operator[](unsigned int l) const { return layers_[l]; }
		layer& operator[](unsigned int l) { return layers_[l]; }
		unsigned int size() const { return static_cast<unsigned int>(layers_.size()); }

		// set a weight
		
		
		// set the weights for a given i neuron (all the j's)

		// set the weights for a given j neuron (everything that contributes to j from previous layer)

		void reset()
		{
			std::default_random_engine rd;
			std::mt19937 eng(rd());
			std::uniform_real_distribution<double> dist(0.0, 1.0);
			for (unsigned int l = 0; l < layers_.size(); ++l)
			{
				for (unsigned int n = 0; n < layers_[l].weights().size(); ++n)
					layers_[l].weights()[n] = dist(eng);
				layers_[l].bias(dist(eng));
			}
		}

		
	private:
		std::vector<layer> layers_;
	};
	
	namespace feedForward
	{
		// transpose
		matrix transpose(const matrix& m)
		{
			matrix result(m.n(), m.m());
			for (unsigned int i = 0; i < m.n(); ++i)
				for (unsigned int j = 0; j < m.m(); ++j)
					result.value(i, j) = m.value(j, i);
			return result;
		}

		// multiplyz
		matrix multiply(const matrix& a, const matrix& b)
		{
			assert(a.n() == b.m());
			matrix result(a.m(), b.n());
     
            for (unsigned int i = 0; i < a.m(); ++i)
                for (unsigned int j = 0; j < b.n(); ++j)
                {
                    double& resultValue = result.value(i, j);
                    for (unsigned int k = 0; k < b.m(); ++k)
                        resultValue += a.value(i, k) * b.value(k, j); 
                    
                }
                        
			return result;
		}

		// add
		matrix add(const matrix& a, const matrix& b)
		{
			assert(a.m() == b.m());
			assert(a.n() == b.n());

			matrix result = a;
			for (unsigned int i = 0; i < result.size(); ++i)
				result[i] += b[i];
			return result;
		}

		matrix add(const matrix& m, double v)
		{
			matrix result = m;
			for (unsigned int i = 0; i < result.size(); ++i)
				result[i] += v;
			return result;
		}

		// a = sigma(z)
		neurons a(const neurons& neu, activation::activationFn f)
		{
			neurons result = neu;
			for (unsigned int i = 0; i < result.size(); ++i)
				result[i] = f(neu[i]);
			return result;
		}

		// z = [a(l) * w(l+1) + b(l+1)]
		neurons z(const neurons& neu, const layer& l)
		{
			return neurons(add(multiply(l.weights(), neu), l.bias()).values());
		}

		// a single feed forward observation...
		output observation(const network& net, const input& input)
		{
			assert(input.weights().m() == 1);
			assert(input.weights().size() == net[0].weights().n());

			neurons neu = input.weights();
			for (unsigned int l = 0; l < (net.size()-1); ++l)
				neu = a(z(neu, net[l]), activation::Fn[net[l + 1].activation()]);

			return neu;
		}
		
		//matrix cost(network& net, const matrix& output)
		//{

		//}

		//void backPropagate(network& net, const matrix& cost)
		//{
		//	//assert(input)
		//	
		//	// calculate error for output layer...

		//	// calculate the error for each subsequent layer...
		//	


		//}

		
	}

}


//#include <fstream>
//#include <vector>
//#include <list>
//#include <functional>
//#include <random>
//#include <numeric>
//#include <assert.h>
//
//namespace neurosys
//{
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//	typedef double neuron;
//
//	namespace activation
//	{
//		typedef std::function<neuron(const neuron& x)> fn;
//
//		static fn linear = [](const neuron& x) { return x; };
//		static fn ReLU = [](const neuron& x) { return x > 0 ? x : 0; }; 
//		static fn leakyReLU = [](const neuron& x) { return x > 0 ? x : 0.01 * x; };
//		static fn fastSigmoid = [](const neuron& x) { return x / (1.0+abs(x)); };
//		static fn sigmoid= [](const neuron& x) { return 1.0 / ( 1.0 + exp(-x)); };
//	}
//
//	struct layer
//	{
//	public:
//		layer(const std::vector<neuron>& neurons, activation::fn activation) : neurons_(neurons), activation_(activation), bias_(0) {}
//		layer(const std::vector<neuron>& neurons, activation::fn activation, double bias) : neurons_(neurons), activation_(activation), bias_(bias) {}
//		layer(unsigned int size, activation::fn activation) : neurons_(size, 0), activation_(activation), bias_(0) {}
//		layer(unsigned int size, activation::fn activation, double bias) : neurons_(size, 0), activation_(activation), bias_(bias) {}
//		
//		layer& operator=(const layer& rhs)
//		{
//			neurons_ = rhs.neurons_;
//			activation_ = rhs.activation_;
//			bias_ = rhs.bias_;
//			return *this;
//		}
//
//		bool operator==(const layer& rhs) const
//		{
//			return neurons_ == rhs.neurons_ && bias_ == rhs.bias_; // also check activation functions are the same?
//		}
//		bool operator!=(const layer& rhs) const
//		{
//			return !(*this == rhs);
//		}
//
//		std::size_t largest()
//		{
//			return std::distance(neurons_.begin(), std::max_element(neurons_.begin(), neurons_.end()));
//		}
//
//		neuron sum()
//		{
//			return std::accumulate(neurons_.begin(), neurons_.end(), 0.0);
//		}
//
//		layer squaredError(const layer& expected) const
//		{
//			// assert layers have same number of items...
//			assert(neurons_.size() == expected.neurons_.size());
//
//			layer result = expected;
//			for (unsigned int n = 0; n < neurons_.size(); ++n)
//				result.neurons_[n] = (result.neurons_[n] - neurons_[n]) * (result.neurons_[n] - neurons_[n]);
//			return result;
//		}
//
//		double sumSquaredError(const layer& expected) const
//		{
//			layer result = squaredError(expected);
//			return result.sum() / static_cast<double>(neurons_.size());
//		}
//
//		std::vector<neuron> neurons_;
//		activation::fn activation_;
//		double bias_;
//	};
//
//	struct network
//	{
//		network(const std::vector<layer>& layers) : layers_(layers)
//		{
//			assert(layers_.size() >= 2);
//
//			// construct the weight matrices...
//			weights_.reserve(layers_.size() - 1);
//			for (unsigned int l = 0; l < layers_.size() - 1; ++l)
//				weights_.push_back(std::vector<neuron>(layers_[l].neurons_.size() * layers_[l+1].neurons_.size()));
//		}
//
//		void reset()
//		{
//			std::default_random_engine rd;
//			std::mt19937 eng(rd());
//			std::uniform_real_distribution<double> dist(0.0, 1.0);
//			for (unsigned int l = 0; l < layers_.size(); ++l)
//			{
//				for (unsigned int n = 0; n < layers_[l].neurons_.size(); ++n)
//					layers_[l].neurons_[n] = dist(eng);
//				layers_[l].bias_ = dist(eng);
//			}
//
//			for (unsigned int w = 0; w < weights_.size(); ++w)
//				for (unsigned int n = 0; n < weights_[w].size(); ++n)
//					weights_[w][n] = dist(eng);
//		}
//
//		// l1 = layer, l1n = layer1 neuron, l2n = layer2 neuron
//		neuron& weight(unsigned int l1, unsigned int l1n, unsigned int l2n)
//		{
//			assert(l1 < layers_.size() - 1);
//			assert(l1n < layers_[l1].neurons_.size());
//			assert(l2n < layers_[l1 + 1].neurons_.size());
//
//			return weights_[l1][(layers_[l1].neurons_.size() * l2n) + l1n];
//		}
//
//		// l1 = layer, l1n = layer1 neuron, l2n = layer2 neuron
//		const neuron& weight(unsigned int l1, unsigned int l1n, unsigned int l2n) const
//		{
//			assert(l1 < layers_.size() - 1);
//			assert(l1n < layers_[l1].neurons_.size());
//			assert(l2n < layers_[l1 + 1].neurons_.size());
//
//			return weights_[l1][(layers_[l1].neurons_.size() * l2n) + l1n];
//		}
//
//		// l1 = layer, l1n = layer1 neuron, l1weights = weights for l1 neuron
//		void weights(unsigned int l1, unsigned int l1n, const std::vector<neuron>& l1nweights)
//		{
//			assert(l1 < layers_.size() - 1);
//			assert(l1n < layers_[l1].neurons_.size());
//			assert(l1nweights.size() == layers_[l1 + 1].neurons_.size());
//
//			std::size_t l1size = layers_[l1].neurons_.size();
//			
//			for (std::size_t n = 0; n < l1nweights.size(); ++n)
//				weights_[l1][(n * l1size) + l1n] = l1nweights[n];
//		}
//
//		// given a neuron, what are its weights to all neurons of the next layer...
//		std::vector<neuron*> forwardWeights(unsigned int l1, unsigned int l1n)
//		{
//			// assert not last layer...
//			assert(l1 != layers_.size());
//			assert(l1n < layers_[l1].neurons_.size());
//			
//			std::size_t l1size = layers_[l1].neurons_.size();
//			std::size_t l2size = layers_[l1 + 1].neurons_.size();
//
//			std::vector<neuron*> result(l2size);
//			for (std::size_t n = 0; n < l2size; ++n)
//				result[n] = &weights_[l1][(l1size * n) + l1n] ;
//			return result;
//		}
//
//		// given a neuron, what are all the weights from the previous layer to the neuron...
//		std::vector<neuron*> backWeights(unsigned int l2, unsigned int l2n)
//		{
//			// assert not first layer...
//			assert(l2 > 0);
//			assert(l2n < layers_[l2-1].neurons_.size());
//
//			std::size_t l1size = layers_[l2 - 1].neurons_.size();
//			std::vector<neuron*> result(l1size);
//			
//			for (std::size_t n = 0; n < l1size; ++n)
//				result[n] = &weights_[l2 - 1][(l1size * l2n) + n];
//			return result;
//		}
//
//		std::vector<layer> layers_;
//		std::vector<std::vector<neuron>> weights_;
//	};
//
//
//	namespace feedForward
//	{
//		neuron dot(const std::vector<neuron>& a, unsigned int aStart, const std::vector<neuron>& b)
//		{
//			neuron result = 0;
//			for (unsigned int i = aStart, j = 0; j < b.size(); ++i, ++j)
//				result += a[i] * b[j];
//			return result;
//		}
//
//		layer feed(const layer& currentLayer, const layer& nextLayer, const std::vector<neuron>& weights)
//		{
//			layer result = nextLayer;
//			for (unsigned int r = 0; r < result.neurons_.size(); ++r)
//				result.neurons_[r] = result.activation_(dot(weights, r * static_cast<unsigned int>(currentLayer.neurons_.size()), currentLayer.neurons_) + nextLayer.bias_);
//			return result;
//		}
//
//		layer observation(const network& n, const layer& input)
//		{
//			layer result = input;
//			for (unsigned int hl = 1; hl < n.layers_.size(); ++hl)
//				result = feed(result, n.layers_[hl], n.weights_[hl-1]);
//			return result;
//		}
//	}
//	
//
//	// calculate error...
//
//	// calculate cost...
//
//	// backPropagation
//	namespace backPropagation
//	{
//		// back propagation of weights...
//		std::vector<neuron> backPropagate(const network& n, unsigned int l2, unsigned int l1, const layer& result, const layer& expected)
//		{
//
//		}
//
//		std::vector<neuron> backPropagate(const network& n, unsigned int l2, unsigned int l1, const layer& result, const layer& expected)
//		{
//
//		}
//
//
//		// for each layer going backwards...
//		/*layer propagate(const network& n, unsigned int l2, unsigned int l1, const layer& result, const layer& expected)
//		{
//
//
//
//			return layer();
//		}*/
//
//
//
//	}
//
//
//}

#endif // NEUROSYS_HPP

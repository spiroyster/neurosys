#ifndef NEUROSYS_HPP
#define NEUROSYS_HPP

#include <vector>
#include <functional>
#include <assert.h>
#include <random>
#include <numeric>
#include <algorithm>

namespace neurosys
{
	class matrix
	{
	public:
		matrix(unsigned int m, unsigned int n) : values_(m * n, 0), stride_(n) {}
		matrix(const std::vector<double>& values, unsigned int stride) : values_(values), stride_(stride) {}

		// construct matrix whose row values are defined by each row of a vector....
		matrix(const matrix& values, unsigned int stride) : values_(values.size() * stride), stride_(stride)
		{
			for (unsigned int r = 0; r < values.size(); ++r)
				for (unsigned int c = 0; c < stride_; ++c)
					values_[r * stride_ + c] = values[r];
		}

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
	
	// matrix specialisation of a single column vector.
	class neurons : public matrix
	{
	public:
		neurons(const matrix& m) : matrix(m.values(), m.n()) {}
		neurons(unsigned int n) : matrix(n, 1) {}
		neurons(const std::vector<double>& values) : matrix(values, 1) {}

		const double& value(unsigned int n) const { return values_[n]; }
		double& value(unsigned int n) { return values_[n]; }

	};

	namespace maths
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

		// multiply
		matrix product(const matrix& a, const matrix& b)
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

		matrix scale(const matrix& m, double scalar)
		{
			matrix result = m;
			for (unsigned int r = 0; r < result.size(); ++r)
				result[r] *= scalar;
			return result;
		}


		matrix hadamard(const matrix& a, const matrix& b)
		{
			assert(a.m() == b.m());
			assert(a.n() == b.n());

			matrix result = a;
			for (unsigned int r = 0; r < result.size(); ++r)
				result[r] *= b[r];
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

		matrix subtract(const matrix& a, const matrix& b)
		{
			assert(a.m() == b.m());
			assert(a.n() == b.n());

			matrix result = a;
			for (unsigned int i = 0; i < result.size(); ++i)
				result[i] -= b[i];
			return result;
		}

		double sum(const matrix& m)
		{
			return std::accumulate(m.values().begin(), m.values().end(), 0.0);
		}

		unsigned int largest(const matrix& m)
		{
			return static_cast<unsigned int>(std::distance(m.values().begin(), std::max_element(m.values().begin(), m.values().end())));
		}

		double mean(const matrix& m)
		{
			double sumResult = sum(m);
			return sumResult == 0 ? 0 : sum(m) / static_cast<double>(m.size());
		}
	}

	namespace activation
	{
		enum function
		{
			sigmoid,
			linear,
			softMax
		};

		typedef std::function<neurons(const neurons&)> activationFn;

		std::vector<activationFn> Fn
		{
			[](const neurons& a) 
			{
				neurons result = a;
				for (unsigned int i = 0; i < result.size(); ++i)
					result[i] = 1.0 / (1.0 + exp(-result[i])); 
				return result;
			},
			[](const neurons& a)
			{ 
				return a; 
			},
			[](const neurons& a)
			{ 
				neurons result = a;
				for (unsigned int i = 0; i < result.size(); ++i)
					result[i] = exp(a[i]);
				double sum = maths::sum(result);
				for (unsigned int i = 0; i < result.size(); ++i)
					result[i] /= sum;
				return result;
			}
		};

		std::vector<activationFn> FnPrime
		{
			[](const neurons& z) 
			{ 
				neurons result = z;
				for (unsigned int i = 0; i < result.size(); ++i)
					result[i] = result[i] * (1.0 - result[i]); 
				return result;
			},
			[](const neurons& z) 
			{ 
				return neurons(std::vector<double>(z.size(), 1.0)); 
			},
			[](const neurons& z) 
			{ 
				neurons result = z;
				for (unsigned int i = 0; i < result.size(); ++i)
					result[i] = result[i] * (1.0 - result[i]); 
				return result;
			}
		};

	}

	// loss function is on neurons... cost function is entire network...

	namespace loss
	{
		
		enum function
		{
			squaredError,
			crossEntropy
		};

		typedef std::function<neurons(const neurons&, const neurons&)> lossFn;
		typedef std::function<double(const neurons&)> costFn;

		std::vector<lossFn> Fn
		{
			[](const neurons& output, const neurons& expected) 
			{
				neurons result(maths::subtract(output, expected).values()); 
				return maths::scale(maths::hadamard(result, result), 0.5); 
			},
			[](const neurons& output, const neurons& expected) 
			{ 
				neurons result = output;
				for (unsigned int i = 0; i < result.size(); ++i)
					result[i] = -(expected[i] * std::log(result[i])) + ((1 - expected[i]) * std::log(1 - result[i]));
				return result;
			}
		};

		std::vector<lossFn> FnPrime
		{
			[](const neurons& output, const neurons& expected) 
			{ 
				return maths::subtract(output, expected); 
			},
			[](const neurons& output, const neurons& expected)
			{
				neurons result = output;
				for (unsigned int i = 0; i < result.size(); ++i)
					result[i] = -(expected[i] / result[i]) + ((1 - expected[i]) / (1 - result[i]));
				return result;
			}
		};

		std::vector<costFn> FnCost
		{
			[](const neurons& error) 
			{ return maths::mean(error); },
			[](const neurons& error) { return -maths::mean(error); }
		};

	}


	// layer (weight matrix), bias...
	class layer
	{
	public:
		layer(unsigned int neuronCount, const activation::function& a, double bias)
			: weights_(neuronCount, 1), bias_(bias), activation_(a)
		{
		}
		layer(unsigned int neuronCount, const activation::function& a)
			: weights_(neuronCount, 1), bias_(1.0), activation_(a)
		{
		}
				
		layer(const matrix& weights, const activation::function& a, double bias) 
			: weights_(weights), bias_(bias), activation_(a) 
		{
		}

		const matrix& weights() const { return weights_; }
		matrix& weights() { return weights_; }

		double bias() const { return bias_; }
		const activation::function& activation() const { return activation_; }
		
		void bias(double b) { bias_ = b; }
		void activation(const activation::function& a) { activation_ = a; }
		
		unsigned int size() const { return weights_.m(); }

		const double& weight(unsigned int i, unsigned int j) const { return weights_.value(j, i); };
		double& weight(unsigned int i, unsigned int j) { return weights_.value(j, i); };


	protected:
		matrix weights_;
		double bias_;
		activation::function activation_;
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

		const neurosys::neurons& neurons() const { return static_cast<const neurosys::neurons&>(weights_); }
	};

	class output : public layer
	{
	public:
		output(unsigned int neuronCount, const activation::function& a, double bias)
			: layer(neuronCount, a, bias)
		{
		}

		output(const neurons& neus)
			: layer(neus, activation::linear, 1.0)
		{
		}

		const neurosys::neurons& neurons() const { return static_cast<const neurosys::neurons&>(weights_); }
	};

	// An artificial neural network, each layer holds a matrix representing the weights....
	// layer 0 is the input, and (no assoctiated weight matrix). The last set of neurons is the output, and the matrix to use to get there.
	class network
	{
	public:
		network(const input& i, const std::vector<layer>& hidden, const output& o)
		{
			if (hidden.empty())
               layers_ = { i, layer(matrix(o.size(), i.size()), o.activation(), o.bias()) };
            else
			{
				layers_.reserve(hidden.size() + 2);
                
                layers_.push_back(i);
				for (unsigned int h = 0; h < (hidden.size()); ++h)
					layers_.push_back(layer(matrix(hidden[h].size(), layers_.back().size()), hidden[h].activation(), hidden[h].bias()));
				    
				layers_.push_back(layer(matrix(o.size(), layers_.back().size()), o.activation(), o.bias()));
			}
		}

		const layer& operator[](unsigned int l) const { return layers_[l]; }
		layer& operator[](unsigned int l) { return layers_[l]; }
		
		// layer count...
		unsigned int size() const { return static_cast<unsigned int>(layers_.size()); }

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

	typedef std::vector<neurons> observation;

	// Perform a single observation...
	observation feedForward(const network& net, const input& i)
	{
		// check input is correct for network...
		assert(net.size() > 2);
		assert(net[0].size() == i.size());

		observation result;
		result.reserve(net.size());
		result.push_back(activation::Fn[net[0].activation()](i.neurons()));

		for (unsigned int l = 1; l < net.size(); ++l)
		{
			// z = [a(l) * w(l+1) + b(l+1) l) ...
			neurosys::matrix z = maths::add(maths::product(net[l].weights(), static_cast<neurosys::neurons>(result.back())), net[l].bias());

			// a = sigma(z) ... Activate the neurons in this layer...
			result.push_back(activation::Fn[net[l].activation()](z));
		}
		return result;
	}

	// back propagate should take an observation...
	network backPropagate(const network& net, const observation& obs, const neurons& error, double learningRate)
	{
		// check observation are correct for network...
		assert(net.size() == obs.size());
		for (unsigned int n = 0; n < net.size(); ++n)
			assert(net[n].size() == obs[n].size());

		network result = net;

		// Calculate the output error... this is the output vs expected. aka dO (delta output)
		neurons dO = error;

		// Calculate dZ (delta sigmoid). this is the derivative of the error, which is denoted by the loss function.
		matrix dZ = activation::FnPrime[net[net.size() - 1].activation()](obs.back());

		// Calculate the (delta output x delta z)... aka local gradient.
		matrix delta = maths::hadamard(dO, dZ);

		for (unsigned int l = net.size() - 1; l > 0; --l)
		{
			// Calculate the delta weight matrix...
			result[l].weights() = maths::product(delta, maths::transpose(obs[l - 1]));
			result[l].bias(maths::mean(delta));

			// calculate the next delta output... dE(l) -> dE(l-1)
			dO = neurons(maths::product(maths::transpose(net[l].weights()), delta).values());

			// calculate the new deltaZ... dZ(l) -> dZ(l-1)
			dZ = activation::FnPrime[net[l-1].activation()](obs[l-1]);

			// And now we can calculate the new dOdZ
			delta = maths::hadamard(dO, dZ);
		}

		// update all the weights and bias...
		for (unsigned int l = net.size() - 1; l > 0; --l)
		{
			result[l].weights() = maths::subtract(net[l].weights(), maths::scale(result[l].weights(), learningRate));
			result[l].bias(net[l].bias() - (result[l].bias() * learningRate));
		}

		return result;
	}
	
	typedef std::function<void(double cost)> iterationComplete;
	
	network epoch(const network& net, const std::vector<input>& inputs, const std::vector<output>& expected, 
		const loss::function& L, double learningRate, iterationComplete iterationCompleteCallback)
	{
		assert(inputs.size() == expected.size());

		// accumulate the error...
		observation obs = feedForward(net, inputs.front());
		neurons err = loss::Fn[L](obs.back(), expected.front().neurons());

		for (unsigned int i = 1; i < inputs.size(); ++i)
		{
			observation iteration = feedForward(net, inputs[i]);

			// accumulate the observation...
			for (unsigned int o = 0; o < iteration.size(); ++o)
				obs[o] = maths::add(obs[o], iteration[o]);

			// accumulate the error...
			err = maths::add(err, loss::Fn[L](iteration.back(), expected[i].neurons()));

			// callback iteration complete...
			iterationCompleteCallback(loss::FnCost[L](err));
		}

		for (unsigned int o = 0; o < obs.size(); ++o)
			for (unsigned int n = 0; n < obs[o].size(); ++n)
				obs[o][n] /= inputs.size();
		for (unsigned int n = 0; n < err.size(); ++n)
				err[n] /= inputs.size();
		
		return backPropagate(net, obs, err, learningRate);
	}

	network epoch(const network& net, const std::vector<input>& inputs, const std::vector<output>& expected, 
		const loss::function& L, double learningRate)
	{
		return epoch(net, inputs, expected, L, learningRate, [](double){});
	}

	// batch... mini batch, sgd etc...

}


#endif // NEUROSYS_HPP

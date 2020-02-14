#ifndef NEUROSYS_HPP
#define NEUROSYS_HPP

#include <vector>
#include <functional>
#include <assert.h>
#include <random>
#include <numeric>


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
			[](const double& x) { return x * (1.0 - x); },
			[](const double& x) { return 1.0; }
		};
	}

	namespace cost
	{
		typedef std::function<double(const double& output, const double& expected)> costFn;

		enum cost
		{
			squaredError,
			difference
		};

		std::vector<costFn> Fn
		{
			[](const double& output, const double& expected) { return 0.5 * ((output - expected) * (output - expected)); },
			[](const double& output, const double& expected) { return output - expected; }
		};

		std::vector<costFn> FnPrime
		{
			[](const double& output, const double& expected) { return (output - expected); },
			[](const double& output, const double& expected) { return 1.0; }
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

		// multiplyz
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

		double average(const matrix& m)
		{
			return sum(m) / static_cast<float>(m.size());
		}
	}

	


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
		
		unsigned int size() const { return weights_.m(); }

		const double& weight(unsigned int i, unsigned int j) const { return weights_.value(j, i); };
		double& weight(unsigned int i, unsigned int j) { return weights_.value(j, i); };


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

		output(const neurons& neus)
			: layer(neus, activation::linear, 1.0)
		{
		}

		const double& neuron(unsigned int n) const { return weights_[n]; }
	};

	
	// network
		// multiple layers..

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
	
	namespace feedForward
	{
		

		// a = sigma(z)
		neurons a(const neurons& ns, activation::activationFn f)
		{
			neurons result = ns;
			for (unsigned int i = 0; i < result.size(); ++i)
				result[i] = f(ns[i]);
			return result;
		}

		// z = [a(l) * w(l+1) + b(l+1) l)
		neurons z(const neurons& ns, const layer& l)
		{
			return neurons(maths::add(maths::product(l.weights(), ns), l.bias()).values());
		}

		neurons traverse(const layer& l, const neurons& i)
		{
			assert(i.n() == 1);
			assert(i.m() == l.weights().n());

			return a(z(i.values(), l), activation::Fn[l.activation()]);
		}

		std::vector<neurons> observation(const network& net, const input& i)
		{
			std::vector<neurons> result;
			result.reserve(net.size());

			result.push_back(i.weights().values());
			for (unsigned int l = 1; l < net.size(); ++l)
				result.push_back(traverse(net[l], result.back()));

			return result;
		}
		
		neurons cost(const output& result, const output& expected, cost::costFn& C)
		{
			matrix ret = result.weights();
			for (unsigned int r = 0; r < result.size(); ++r)
				ret[r] = C(result.weights()[r], expected.weights()[r]);
			return neurons(ret.values());
		}

		// cost is (o - t).
		network backPropagation(const network& net, const input& i, const output& expected, const cost::cost& C, const double& learningRate)
		{
			assert(net.size() >= 2);

			network result = net;

			// perform the forward propagation...
			std::vector<neurons> ff = observation(net, i);
			
			// Calculate the output error... this is the output vs expected.
			matrix dO = cost(ff.back(), expected, cost::FnPrime[C]);

			// Calculate dZ (delta sigmoid)
			matrix dZ = a(neurons(dO.values()), activation::FnPrime[net[net.size() - 1].activation()]);

			// Calculate the delta matrix.. (dO * dZ)
			matrix dZdO = maths::hadamard(dO, dZ);
			
			for (unsigned int l = net.size() - 1; l > 0; --l)
			{
				// calculate the delta weight matrix... (dZ(l) * h(l-1))
				matrix dW = maths::product(dZdO, maths::transpose(ff[l - 1]));

				// update the weights... wn = wn - learnRate ( dW )
				result[l].weights() = maths::subtract(net[l].weights(), maths::scale(dW, learningRate));

				// update the bias... sum of dZ (or just dZ)
				result[l].bias(net[l].bias() - (learningRate * maths::sum(dZdO)));

				// calculate the next delta... dE(l) -> dE(l-1)
				dO = maths::product(maths::transpose(net[l].weights()), dZdO);

				// calculate the new deltaZ... dZ(l) -> dZ(l-1)
				dZ = a(neurons(dO.values()), activation::FnPrime[net[l-1].activation()]);

				dZdO = maths::hadamard(dO, dZ);
			}

			return result;
		}

		typedef std::function<neurons(unsigned int n, const output& o)> expectedFn;

		network train(const network& net, const std::vector<input>& i, const std::vector<output>& o, const cost::cost& C, double learningRate, unsigned int start, unsigned int end)
		{


			for (unsigned int n = start; n < end; ++n)
			{


			}

			// back propagate...

		}

		network train(const network& net, const std::vector<input>& i, const std::vector<output>& o, const cost::cost& C, double learningRate, unsigned int batchSize)
		{
			network result = net;
			for (unsigned int n = 0; n < i.size(); n += batchSize)
			{
				result = train(result, i, o, C, learningRate, n )
			}
				
		}

		network train(const network& net, const std::vector<input>& i, const std::vector<output>& o, const cost::cost& C, double learningRate)
		{
			return train(net, i, o, C, learningRate, i.size());
		}

	}

}


#endif // NEUROSYS_HPP

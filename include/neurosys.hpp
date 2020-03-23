#ifndef NEUROSYS_HPP
#define NEUROSYS_HPP

#include <vector>
#include <functional>
#include <assert.h>
#include <random>
#include <numeric>
#include <algorithm>

#define NEUROSYS_COUT

#ifdef NEUROSYS_COUT
#include <iostream>
#endif // NEUROSYS_COUT

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
			return sum(m) / static_cast<double>(m.size());
		}

		double median(const matrix& m)
		{
			std::vector<double> sorted = m.values();
			std::sort(sorted.begin(), sorted.end());
			return sorted[(m.size() + 1) / 2];
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
					result[i] = exp(result[i]);
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

	namespace cost
	{
		
		enum function
		{
			squaredError,
			crossEntropy
		};

		typedef std::function<neurons(const neurons&, const neurons&)> costFn;
		typedef std::function<double(const neurons&, const neurons&)> networkCostFn;

		std::vector<costFn> Fn
		{
			[](const neurons& output, const neurons& expected) 
			{
				neurons result(maths::subtract(expected, output).values()); 
				return maths::scale(maths::hadamard(result, result), 0.5); 
			},

			[](const neurons& output, const neurons& expected) 
			{ 
				neurons result = output;
				for (unsigned int i = 0; i < result.size(); ++i)
					result[i] = -expected[i] * std::log(result[i]);
				return result;
			}
		};

		std::vector<costFn> FnPrime
		{
			[](const neurons& output, const neurons& expected) { return maths::subtract(output, expected); },
			[](const neurons& output, const neurons& expected) { return maths::subtract(expected, output); }
		};

		std::vector<networkCostFn> FnNetwork
		{
			[](const neurons& output, const neurons& expected) { return maths::mean(Fn[function::squaredError](output, expected)); },
			[](const neurons& output, const neurons& expected) { return maths::mean(Fn[function::crossEntropy](output, expected)); },
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

	void shuffle(std::vector<input>& i, std::vector<output>& o)
	{
		// Shuffle the items... apply same shuffling to input (dataset) and output (labels)

	}

	// An artificial neural network, each layer holds a matrix representing the weights....
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

	// Perform a single observation... return the un-activated neurons
	std::vector<neurons> feedForward(const network& net, const input& i)
	{
		std::vector<neurons> result;
		result.reserve(net.size());

		result.push_back(i.neurons());
		for (unsigned int l = 1; l < net.size(); ++l)
		{
			// z = [a(l) * w(l+1) + b(l+1) l) ... N.B Perform the activation of this layer before calcing z.
			neurons z = neurons(maths::add(maths::product(net[l].weights(), activation::Fn[net[l].activation()](result.back())), net[l].bias()).values());

			// a = sigma(z) ... Activate the neurons in this layer...
			result.push_back(z);
		}
		return result;
	}

	output observation(const network& net, const input& i)
	{
		return activation::Fn[net[net.size() - 1].activation()](feedForward(net, i).back());
	}

	network backPropagate(const network& net, const input& i, const output& expected, const cost::function& C, double learningRate)
	{
		assert(net.size() >= 2);

		network result = net;

		// first perform the feedforward... (this gives us z's of the feed forward)...
		std::vector<neurons> ff = feedForward(net, i);

		// Calculate the output error... this is the output vs expected. aka dO (delta output)
		output o = activation::Fn[net[net.size() - 1].activation()](ff.back());
		neurons dO = cost::FnPrime[C](o.neurons(), expected.neurons());

		// Calculate dZ (delta sigmoid)
		matrix dZ = activation::FnPrime[net[net.size() - 1].activation()](ff.back());

		// Calculate the dOdZ (delta output x delta z)... aka local gradient.
		matrix dOdZ = maths::hadamard(dO, dZ);

		for (unsigned int l = net.size() - 1; l > 0; --l)
		{
			// Calculate the delta weight matrix...
			result[l].weights() = maths::product(dOdZ, maths::transpose(ff[l - 1]));
			result[l].bias(maths::sum(dOdZ));

			// calculate the next delta output... dE(l) -> dE(l-1)
			dO = neurons(maths::product(maths::transpose(net[l].weights()), dOdZ).values());

			// calculate the new deltaZ... dZ(l) -> dZ(l-1)
			dZ = activation::FnPrime[net[l-1].activation()](ff[l-1]);

			// And now we can calculate the new dOdZ
			dOdZ = maths::hadamard(dO, dZ);
		}

		// update all the weights and bias...
		for (unsigned int l = net.size() - 1; l > 0; --l)
		{
			result[l].weights() = maths::subtract(net[l].weights(), maths::scale(result[l].weights(), learningRate));
			result[l].bias(net[l].bias() - (result[l].bias() * learningRate));
		}

		return result;
	}


	unsigned int test(const network& net, const std::vector<input>& tests, const std::vector<output>& expected, 
		std::function<unsigned int(const input& i, const output& o, const output& expected)> predicate)
	{
		unsigned int correct = 0;

		#ifdef NEUROSYS_COUT
			std::cout << "testing: [";
			unsigned int pbCurrent = 0;
		#endif
		
		#pragma omp parallel for
		for (int n = 0; n < tests.size(); ++n)
		{
			output result = observation(net, tests[n]);//feedForward(net, tests[n]).back();
			unsigned int score = predicate(tests[n], result, expected[n]);

			#pragma omp critical
			{
				correct += score;
				#ifdef NEUROSYS_COUT
					unsigned int progress = static_cast<unsigned int>(static_cast<double>(n) * 20.0/static_cast<double>(tests.size()));
					if (pbCurrent != progress)
					{
						std::cout << "=";
						std::cout.flush();
						pbCurrent = progress;
					}
				#endif		
			}
		}

		#ifdef NEUROSYS_COUT
			std::cout << "] " << correct << "/" << tests.size() << " correct (" << 
				static_cast<double>(correct) * 100.0 / static_cast<double>(tests.size()) << "%)";
		#endif
		
		return correct;
	}

	

	typedef std::function<bool(const network& net, double networkCose)> TrainBatchFinish;

	network train(const network& net, const std::vector<input>& training, const std::vector<output>& expected, 
		const cost::function& C, double learningRate, unsigned int start, unsigned int end)
	{
		#ifdef NEUROSYS_COUT
			std::cout << "training " << start << "|" << end << " (" << training.size() << ") [";
			unsigned int pbCurrent = 0;
		#endif

		if (end > training.size())
			end = static_cast<unsigned int>(training.size());

		input i(training.front().size());
		output e(expected.front().size());

		#pragma omp parallel for
		for (int n = start; n < static_cast<int>(end); ++n)
		{
			std::vector<neurons> ff = feedForward(net, training[n]);

			#pragma omp critical
			{
				// Accumulate...
				i.weights() = maths::add(i.neurons(), training[n].neurons());
				e.weights() = maths::add(e.neurons(), expected[n].neurons());

				#ifdef NEUROSYS_COUT
					unsigned int progress = static_cast<unsigned int>(static_cast<double>(n) * 20.0/static_cast<double>(end - start));
					if (pbCurrent != progress)
					{
						std::cout << "=";
						std::cout.flush();
						pbCurrent = progress;
					}
				#endif		
			}
		}

		double scalar = 1.0 / end - start;
		i.weights() = maths::scale(i.weights(), scalar);
		e.weights() = maths::scale(e.weights(), scalar);
		
		#ifdef NEUROSYS_COUT
			std::cout << "] C= " << cost::FnNetwork[C](i.neurons(), e.neurons()) << " ";
		#endif
		
		// Backpropagate...
		return backPropagate(net, i, e, C, learningRate);
	}

	network train(const network& net, const std::vector<input>& training, const std::vector<output>& expected, 
		const cost::function& C, double learningRate)
	{
		return train(net, training, expected, C, learningRate, 0, static_cast<unsigned int>(training.size()));
	}

	network train(const network& net, const std::vector<input>& training, const std::vector<output>& expected, 
		const cost::function& C, double learningRate, TrainBatchFinish trainBatchFinishCallback)
	{
		network result = train(net, training, expected, C, learningRate, 0, static_cast<unsigned int>(training.size()));
		trainBatchFinishCallback(result, 42.0);
		return result;	
	}

	
	network train(const network& net, const std::vector<input>& training, const std::vector<output>& expected, 
		const cost::function& C, double learningRate, unsigned int batch, TrainBatchFinish trainBatchFinishCallback)
	{
		network working = net;
		for (unsigned int b = 0; b < training.size(); b += batch)
		{
			working = train(working, training, expected, C, learningRate, b, b + batch);
			trainBatchFinishCallback(working, 42.0);
		}
		return working;
	}

	network train(const network& net, const std::vector<input>& training, const std::vector<output>& expected, 
		const cost::function& C, double learningRate, unsigned int batch)
	{
		return train(net, training, expected, C, learningRate, batch, [](const network& net, double networkCost){ return true; });
	}

}


#endif // NEUROSYS_HPP

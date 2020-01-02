#ifndef NEUROSYS_HPP
#define NEUROSYS_HPP

#include <fstream>
#include <vector>
#include <list>
#include <functional>

namespace neurosys
{

	typedef double cell;


	namespace activation
	{
		typedef std::function<cell(const cell& x)> fn;

		static fn linear = [](const cell& x) { return x; };
		static fn ReLu = [](const cell& x) { return x; };
		static fn sigmoid = [](const cell& x) { return x; };
	}
	

	class layer
	{
	public:
		layer()
			: fn_(activation::linear)
		{
		}

		layer(std::size_t sz, const activation::fn& f)
			:	values_(sz), fn_(f)
		{
		}

		layer(const activation::fn& f, std::size_t sz, std::function<cell(std::size_t n)> initialiser)
			: values_(sz), fn_(f)
		{
			for (std::size_t n = 0; n < sz; ++n)
				values_[n] = initialiser(n);
		}

		layer(const activation::fn& f, const std::vector<cell>& values)
			: values_(values), fn_(f)
		{
		}

		bool operator==(const layer& rhs) const
		{
			return values_ == rhs.values_;
			// need to compare functions ... && fn_ == rhs.fn_;
		}

		cell& operator[](std::size_t n) { return values_[n]; }
		const cell& operator[](std::size_t n) const { return values_[n]; }
		
		std::size_t size() const { return values_.size(); }

		std::vector<cell>::const_iterator begin() const { return values_.begin(); }
		std::vector<cell>::const_iterator end() const { return values_.end(); }

		std::vector<cell>::iterator begin() { return values_.begin(); }
		std::vector<cell>::iterator end() { return values_.end(); }

		const std::vector<cell>& values() const { return values_; }
		std::vector<cell>& values() { return values_; }

		const activation::fn& activationFn() const { return fn_; }

	private:
		std::vector<cell> values_;
		const activation::fn& fn_;
	};


	// model (one or more layers)... first layer is input, last layer is output
	class model
	{
	public:

		model(const std::vector<layer>& layers)
		{
			if (layers.empty())
				throw std::exception("Model requires at least two layers.");

			std::size_t layerCount = layers.size();
			for (unsigned int l = 0; l < (layers.size() - 1); ++l)
				layerCount += layers[l].values().size();
			
			layers_.reserve(layerCount);

			// we need to create the synapse layers...
			for (unsigned int l = 0; l < (layers.size() - 1); ++l)
			{
				// get the current layer...
				layerIndex_.push_back(layers_.size()); 
				layers_.push_back(layers[l]);

				// create a new layer for each current layer neuron
				for (unsigned int ln = 0; ln < layers[l].values().size(); ++ln)
					layers_.push_back(layer(layers[l + 1].values().size(), layers[l].activationFn()));
			}

			// add the last layer...
			layers_.push_back(layers.back());
			layerIndex_.push_back(layers_.size() - 1);
		}

		std::size_t paramterCount() const
		{
			std::size_t result = 0;
			for (unsigned int l = 0; l < layers_.size(); ++l)
				result += layers_[l].values().size();
			return result;
		}

		std::size_t layerCount() const
		{
			return layers_.size();
		}
		
		// get the layer (layerNum relates to original input layer index).
		const layer& operator[](unsigned int layerIndex) const
		{
			return layers_[layerIndex_[layerIndex]];
		}

		layer& operator[](unsigned int layerIndex)
		{
			return layers_[layerIndex_[layerIndex]];
		}

		// get the input layer...
		const layer& input() const
		{
			return layers_.front();
		}

		// get the output layer...
		const layer& output() const
		{
			return layers_.back();
		}

		const layer& synapses(unsigned int layerNum, unsigned int neuron) const
		{
			return layers_[layerIndex_[layerNum] + neuron + 1];
		}

		class iterator
		{
		public:
			iterator() : layerNum_(0), neuron_(0), model_(0), cell_(0) {}

			iterator(unsigned int layerNum, unsigned int neuron, model& m, cell& c)
				: layerNum_(layerNum), neuron_(neuron), model_(&m), cell_(&c)
			{
			}

			bool operator==(const iterator& rhs) const
			{
				return rhs.layerNum_ == layerNum_ && rhs.neuron_ == neuron_ && rhs.model_ == model_ && rhs.cell_ == cell_;
			}
			bool operator!=(const iterator& rhs) const
			{
				return !(*this == rhs);
			}

			cell* operator*() { return cell_; }

			// next neuron in layer...
			iterator down() { return model_->cell(layerNum_, neuron_ + 1); }

			// prev neuron in layer...
			iterator up() { return model_->cell(layerNum_, neuron_ - 1); }

			// next laayer neuron...
			iterator forward(unsigned int neuron) { return model_->cell(layerNum_ + 1, neuron); }

			// prev layer neuron...
			iterator back(unsigned int neuron) { return model_->cell(layerNum_ - 1, neuron); }

			// back layer...

			// forward layer...

			// back synapses
			//std::vector<cell*> forwardSynapses()
			//{
				
			//}
			
			// forward synpases
			std::vector<cell*> backSynapses()
			{
				std::vector<neurosys::cell*> result;
				if (layerNum_ > 0)
				{
					std::size_t startLayer = model_->layerIndex_[layerNum_ - 1];
					std::size_t endLayer = model_->layerIndex_[layerNum_];
					result.reserve(model_->layers_[startLayer].values().size());
					for (std::size_t l = startLayer + 1; l < endLayer; ++l)
						result.push_back(&model_->layers_[l][neuron_]);
				}
				return result;
			}

		private:
			unsigned int layerNum_;
			unsigned int neuron_;
			model* model_;
			cell* cell_;
		};

		iterator cell(unsigned int layerNum, unsigned int neuron)
		{
			if (layerNum < layerIndex_.size())
			{
				layer& lay = (*this)[layerNum];
				if (neuron < lay.values().size())
					return iterator(layerNum, neuron, *this, lay[neuron]);
			}
			return iterator();
		}

		// reset the weights/nodes in the model (layerNum, neuron)
		typedef std::function<neurosys::cell(unsigned int, unsigned int)> resetCallback;

		void reset(resetCallback layerResetFunc, resetCallback synapseResetFunc)
		{
			unsigned int currentLayerNum = 0;
			for (unsigned int l = 0; l < layers_.size(); ++l)
			{
				layer& lay = layers_[l];
				if (l == layerIndex_[currentLayerNum])
				{
					for (unsigned int n = 0; n < lay.size(); ++n)
						lay[n] = layerResetFunc(currentLayerNum, n);
					++currentLayerNum;
				}
				else
				{
					for (unsigned int n = 0; n < lay.size(); ++n)
						lay[n] = synapseResetFunc(currentLayerNum, n);
				}
			}
		}


	private:
		std::vector<layer> layers_;
		std::vector<std::size_t> layerIndex_;
	};
	



}

#endif // NEUROSYS_HPP
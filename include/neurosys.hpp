#ifndef NEUROSYS_HPP
#define NEUROSYS_HPP

#include <fstream>
#include <vector>
#include <list>

namespace neurosys
{
	namespace activation
	{
		enum activation
		{
			custom = 0,
			linear,
			sigmoid,
			ReLu,

			cairoelephant
		};
	}
	
	// the cell payload...
	typedef double payload;

	// a cell...
	typedef std::pair<double, activation::activation> cell;

	// a layer...
	typedef std::vector<cell> layer;

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
				layerCount += layers[l].size();
			
			layers_.reserve(layerCount);

			// we need to create the synapse layers...
			for (unsigned int l = 0; l < (layers.size() - 1); ++l)
			{
				// get the current layer...
				layerIndex_.push_back(layers_.size()); 
				layers_.push_back(layers[l]);

				// create a new layer for each current layer neuron
				for (unsigned int ln = 0; ln < layers[l].size(); ++ln)
					layers_.push_back(layer(layers[l + 1].size(), neurosys::cell(0, layers[l][ln].second)));
			}

			// add the last layer...
			layers_.push_back(layers.back());
			layerIndex_.push_back(layers_.size() - 1);
		}

		std::size_t paramterCount() const
		{
			std::size_t result = 0;
			for (unsigned int l = 0; l < layers_.size(); ++l)
				result += layers_[l].size();
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
					result.reserve(model_->layers_[startLayer].size());
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
				if (neuron < lay.size())
					return iterator(layerNum, neuron, *this, lay[neuron]);
			}
			return iterator();
		}

		//// get the synapse layer (connections between two layers)
		//// synapseForwardItr
		//// current layer cell...

		//class synapseItr
		//{
		//public:

		//	synapseItr(unsigned int layerNum, unsigned int neuron, model& m, cell& synapse)
		//		:	m_(&m), layerNum_(layerNum), neuron_(neuron), synapse_(&synapse)
		//	{
		//	}
		//	
		//	cell& operator*() { return *synapse_; }

		//private:
		//	unsigned int layerNum_;
		//	unsigned int neuron_;
		//	cell* synapse_;
		//	model* m_;
		//};

		//static 

		//synapseItr synapse(unsigned int layerNum, unsigned int neuron, unsigned int synapse)
		//{

		//	if (layerNum < layerIndex_.size())
		//	{
		//		unsigned int address = layerIndex_[layerNum];
		//		if (neuron < layers_[address].size())
		//		{
		//			address += neuron + 1;
		//			if (synapse < layers_[address].size())
		//				return synapseItr(layerNum, neuron, *this, layers_[address][synapse]);
		//		}

		//	}
		//	
		//	//return synapseItr(layerNum, neuron, *this, layers_[layerIndex_[layerNum]][)

		//}
		
		//class synpaseItr
		//{
		//public:


		//private:
		//	synpaseItr forward(unsigned int neuronIndex)
		//	{
		//		synpaseItr newItr = *this;
		//		
		//		// check the next layer...
		//		if (address_.layerNum_ + 1 >= model_.layerIndex_.size())
		//			throw std::exception("Unable to move iterator forward. No layer.");

		//	}

		//	address backward()
		//	{

		//	}
		//	
		//	address address_;
		//	model& model_;
		//};



		/*const layer& synapseForward(unsigned int layerIndex, unsigned int neuron)
		{
			return layers_[layerIndex_[layerIndex] + neuron + 1];
		}*/

		/*std::vector<const cell*> synpaseBackward(unsigned int layerIndex, unsigned int neuron)
		{
			
		}*/

		//// layer num, and the neuron num within that layer...
		//const layer& synapses(unsigned int layerNum, unsigned int neuron) const
		//{
		//	return layers_[layerIndex_[layerNum] + neuron];
		//}


		//const cell& synapseTarget(unsigned int layerNum, unsigned int neuron, unsigned int targetNeuron)
		//{
		//	//return synapses(layerNum, neuron)
		//}
		

		// train 
		layer train(const layer& inputLayerData)
		{
			// take the input data and fire through network...
		}

		// read
		void read(std::ifstream& file)
		{

		}

		// write
		void write(std::ofstream& file)
		{
			// write header...
			file << "NEUROSYS1\n";

			/*
			#layer0
			[A1][A1B1, A1B2, A1B3 ... A1Bn]
			[A2][A2B1, A2B2, A2B3 ... A1Bn]
			[A3][A3B1, A3B2, A3B3 ... A1Bn]
			#layer1
			[B1][B1C1, B1C2 ... B1Cn]
			[B2][B2C1, B2C2 ... B2Cn]
			#layer3
			[C1]
			[C2]
			
			*/

		}

		

	private:
		std::vector<layer> layers_;
		std::vector<std::size_t> layerIndex_;
	};
	



}

#endif // NEUROSYS_HPP
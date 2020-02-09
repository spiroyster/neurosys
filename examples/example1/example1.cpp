#include "../../include/neurosys.hpp"
#include "MNIST.hpp"

#include <iostream>

int main(int argc, char** argv)
{
	// t10k-labels.idx1-ubyte
	// train-labels.idx1-ubyte
	// t10k-images.idx3-ubyte
	// train-images.idx3-ubyte

	// load in the MNIST examples...
	try
	{
		std::cout << "Loading data (t10k-images-idx3-ubyte)...";
		std::vector<neurosys::input> trainingImages = neurosys::MNIST::HandwritingImagesRead("t10k-images-idx3-ubyte");
		//std::vector<neurosys::input> trainingImages = neurosys::MNIST::HandwritingImagesRead("train-images-idx3-ubyte");
		std::cout << "done.\n";

		std::cout << "Loading data (t10k-labels-idx1-ubyte)...";
		std::vector<char> trainingLabels = neurosys::MNIST::HandwritingLabelsRead("t10k-labels-idx1-ubyte");
		//std::vector<char> trainingLabels = neurosys::MNIST::HandwritingLabelsRead("train-labels-idx1-ubyte");
		std::cout << "done.\n";

		// construct our NN... single 16 node hidden layers, output is 10 neurons, one for each possible digit.
		/*neurosys::network net({
			trainingImages.front(),
			neurosys::layer(16, neurosys::activation::fastSigmoid),
			neurosys::layer({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }, neurosys::activation::fastSigmoid)
			});*/
		neurosys::network net(trainingImages.front(), { neurosys::layer(16, neurosys::activation::sigmoid) }, neurosys::output(10, neurosys::activation::sigmoid, 1.0));

		// set to randoms...
		net.reset();

		// put the values through an see what we get!
		//unsigned int correct = 0;
		//for (int i = 0; i < trainingImages.size(); ++i)
		//{
			//std::cout << "forward " << i << " |";

			// put through network...
			//neurosys::layer output = neurosys::feedForward::observation(net, trainingImages[i]);

			// find out which digit...(this is the index of the highest value neruon in the output)
			//std::size_t value = output.largest();

			//std::cout << "  expected: " << static_cast<unsigned int>(trainingLabels[i]) << " actual : " << value << "  (neuron:" << output.neurons_[value] << ")\n";

			// compare...
			//correct += value == trainingLabels[i] ? 1 : 0;
		//}

		// caluclate the percentage correct...
		//double percent = static_cast<double>(correct) / static_cast<double>(trainingImages.size()) * 100.0;

		//std::cout << "==================\n";
		//std::cout << " " << correct << " correct observations (" << percent << "%).\n";

		return 0;
	}
	catch (const std::exception& e)
	{
		std::wcout << "ERR! " << e.what();
	}
	
	return 1;

}

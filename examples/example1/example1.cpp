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
		std::vector<neurosys::input> testImages = neurosys::MNIST::HandwritingImagesRead("t10k-images-idx3-ubyte");		
		std::cout << "done.\n";

		std::cout << "Loading data (t10k-labels-idx1-ubyte)...";
		std::vector<neurosys::output> testLabels = neurosys::MNIST::HandwritingLabelsRead("t10k-labels-idx1-ubyte");
		std::cout << "done.\n";

		neurosys::network net(testImages.front(), { neurosys::layer(16, neurosys::activation::sigmoid) }, neurosys::output(10, neurosys::activation::sigmoid, 1.0));

		// set to randoms...
		net.reset();

		// First see its performance with no training...
		unsigned int correct = 0;
		for (unsigned int n = 0; n < testImages.size(); ++n)
			correct += neurosys::maths::largest(neurosys::feedForward::observation(net, testImages[n]).back()) == neurosys::maths::largest(testLabels[n].weights()) ? 1 : 0;
		
		std::cout << "Untrained model. " << testImages.size() << " observations" << " : " << (static_cast<double>(correct) / static_cast<double>(testImages.size())) * 100.0 << " % correct.\n";

		// Loading in the training data...
		std::cout << "Loading data (train-images-idx3-ubyte)...";
		std::vector<neurosys::input> trainImages = neurosys::MNIST::HandwritingImagesRead("train-images-idx3-ubyte");
		std::cout << "done.\n";

		std::cout << "Loading data (train-labels-idx1-ubyte)...";
		std::vector<neurosys::output> trainLabels = neurosys::MNIST::HandwritingLabelsRead("train-labels-idx1-ubyte");
		std::cout << "done.\n";

		// train the model...
		net = neurosys::feedForward::train(net, trainImages, trainLabels, neurosys::cost::squaredError, 0.01, 32);

		correct = 0;
		for (unsigned int n = 0; n < testImages.size(); ++n)
			correct += neurosys::maths::largest(neurosys::feedForward::observation(net, testImages[n]).back()) == neurosys::maths::largest(testLabels[n].weights()) ? 1 : 0;

		std::cout << "Trained model. " << testImages.size() << " observations" << " : " << (static_cast<double>(correct) / static_cast<double>(testImages.size())) * 100.0 << " % correct.\n";


		// replay the tests.





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

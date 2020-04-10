#include <iostream>

#include "../../include/neurosys.hpp"
#include "MNIST.hpp"

int main(int argc, char** argv)
{
	// load in the MNIST examples...
	try
	{
        // Load in the test data.
		std::cout << "Loading data (t10k-images-idx3-ubyte)...";
		std::vector<neurosys::input> testImages = neurosys::MNIST::HandwritingImagesRead("t10k-images-idx3-ubyte");		
		std::cout << "done.\n";

		std::cout << "Loading data (t10k-labels-idx1-ubyte)...";
		std::vector<neurosys::output> testLabels = neurosys::MNIST::HandwritingLabelsRead("t10k-labels-idx1-ubyte");
		std::cout << "done.\n";

        // Load in the training data...
		std::cout << "Loading data (train-images-idx3-ubyte)...";
		std::vector<neurosys::input> trainImages = neurosys::MNIST::HandwritingImagesRead("train-images-idx3-ubyte");
		std::cout << "done.\n";

		std::cout << "Loading data (train-labels-idx1-ubyte)...";
		std::vector<neurosys::output> trainLabels = neurosys::MNIST::HandwritingLabelsRead("train-labels-idx1-ubyte");
		std::cout << "done.\n";

        // Create our neural network. 
		neurosys::network net(testImages.front(), { neurosys::layer(800, neurosys::activation::sigmoid) }, 
			neurosys::output(10, neurosys::activation::sigmoid, 0.1));

		// Reset our neural network. This sets both the bias and weights to random values.
		net.reset();

		double accuracy = 0;
		while (accuracy < 0.5)
		{
			// train the network....
			net = neurosys::train(net, trainImages, trainLabels, neurosys::loss::function::crossEntropy, 1, 1,
				[&trainImages, &trainLabels, &accuracy](const neurosys::network& net, double cost)
				{
					// test the network...
					unsigned int correct = neurosys::test(net, trainImages, trainLabels,
						[](const neurosys::input& i, const neurosys::output& o, const neurosys::output& expected)
						{
							return neurosys::maths::largest(o.neurons()) == neurosys::maths::largest(expected.neurons());
						});

					accuracy = static_cast<double>(correct)/static_cast<double>(trainImages.size());

					std::cout << "\n";

					// carry on...
					return true;
				});

		}
		
		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << "ERR! " << e.what();
	}
	
	return 1;

}

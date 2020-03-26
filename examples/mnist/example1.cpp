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
		neurosys::network net(testImages.front(), { neurosys::layer(32, neurosys::activation::sigmoid) }, 
			neurosys::output(10, neurosys::activation::sigmoid, 1.0));

		// Reset our neural network. This sets both the bias and weights to random values.
		net.reset();
		while (1==1)
		{
			// train the network....
			net = neurosys::train(net, trainImages, trainLabels, neurosys::cost::function::crossEntropy, 0.01, 1,
				[&testImages, &testLabels](const neurosys::network& net, double cost)
				{
					// test the network...
					neurosys::test(net, testImages, testLabels,
						[](const neurosys::input& i, const neurosys::output& o, const neurosys::output& expected)
						{
							return neurosys::maths::largest(o.neurons()) == neurosys::maths::largest(expected.neurons());
						});


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

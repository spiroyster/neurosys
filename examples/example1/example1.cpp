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
		neurosys::network net(testImages.front(), { neurosys::layer(16, neurosys::activation::sigmoid) }, neurosys::output(10, neurosys::activation::sigmoid, 1.0));

		// Reset our neural network. This sets both the bias and weights to random values.
		net.reset();

		// First check the accuracy of the model without training.
		unsigned int correct = neurosys::MNIST::test(net, testImages, testLabels);
		std::cout << "Untrained. " << correct << " observations: " << (static_cast<double>(correct) / static_cast<double>(testImages.size())) * 100.0 << " % correct.\n";

        // single training...
        
        
        
        
		// training...
        for (unsigned int m = 0; m < trainImages.size(); ++m)
        {
            std::cout << "Training " << m << ". ";
            
            // train the model.
            net = neurosys::feedForward::backPropagate(net, trainImages[m], trainLabels[m], neurosys::cost::squaredError, 0.01);

            // test the new model...
            correct = neurosys::MNIST::test(net, testImages, testLabels);
            
            std::cout << correct << " observations: " << (static_cast<double>(correct) / static_cast<double>(testImages.size())) * 100.0 << " % correct.\n";
        }
         
		return 0;
	}
	catch (const std::exception& e)
	{
		std::wcout << "ERR! " << e.what();
	}
	
	return 1;

}

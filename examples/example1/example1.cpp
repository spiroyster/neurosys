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
		bool singleTraining = false;
		bool batchTraining = false;


		if (singleTraining)
		{
			for (unsigned int m = 0; m < trainImages.size(); ++m)
			{
				std::cout << "Training " << m << ". ";

				// train the model.
				net = neurosys::feedForward::backPropagate(net, trainImages[m], trainLabels[m].neurons(), neurosys::cost::function::squaredError, 0.01);

				// test the new model...
				correct = neurosys::MNIST::test(net, testImages, testLabels);

				std::cout << correct << " observations: " << (static_cast<double>(correct) / static_cast<double>(testImages.size())) * 100.0 << " % correct.\n";
			}
		}
		

		// batch...
		//for (unsigned int m = 0; m < trainImages.size(); ++m)
		//{
		//	std::cout << "Training Batch " << m << ". ";

		//	// train the model.
		//	net = neurosys::feedForward::backPropagate(net, trainImages, trainLabels, neurosys::cost::function::squaredError, 0.01, 32);

		//	// test the new model...
		//	correct = neurosys::MNIST::test(net, testImages, testLabels);

		//	std::cout << correct << " observations: " << (static_cast<double>(correct) / static_cast<double>(testImages.size())) * 100.0 << " % correct.\n";
		//}

		// batch train the model.
		/*unsigned int batchSize = 10;
		unsigned int batchProgress = 0;

		net = neurosys::feedForward::backPropagate(net, trainImages, trainLabels, neurosys::cost::function::squaredError, 0.01, batchSize,
			[&trainImages, &batchSize, &batchProgress](unsigned int b, const neurosys::network& result) 
			{ 
				std::cout << "Training batch #" << b << " (" << batchSize * b << "/" << trainImages.size() << ") |"; 
				batchProgress = 0;
			},
			[&testImages, &testLabels](unsigned int b, const neurosys::network& result) 
			{
				std::cout << " Testing...";
				unsigned int correct = neurosys::MNIST::test(result, testImages, testLabels);
				std::cout << (static_cast<double>(correct) / static_cast<double>(testImages.size())) * 100.0 << "% correct.\n";
			},
			[&batchSize, &batchProgress](unsigned int b, unsigned int n, const neurosys::network& result) 
			{
				unsigned int currentBatchProgress = static_cast<unsigned int>(static_cast<double>(n) * 10.0 / batchSize);
				if (currentBatchProgress != batchProgress)
				{
					std::cout << "=";
					currentBatchProgress = batchProgress;
				}
				if (n == batchSize - 1)
					std::cout << "|";
				
			});*/

		// Training batch #4 (128/60000) |======================| (9.8% correct).


		//for (unsigned int m = 0; m < trainImages.size(); ++m)
		//{
		//	std::cout << "Training Batch " << m << ". ";

		//	
		//	// test the new model...
		//	correct = neurosys::MNIST::test(net, testImages, testLabels);

		//	std::cout << correct << " observations: " << (static_cast<double>(correct) / static_cast<double>(testImages.size())) * 100.0 << " % correct.\n";
		//}
         
		return 0;
	}
	catch (const std::exception& e)
	{
		std::cout << "ERR! " << e.what();
	}
	
	return 1;

}

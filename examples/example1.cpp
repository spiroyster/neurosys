#include "..\include\neurosys.hpp"
#include "..\include\MNIST.hpp"

int main(int argc, char** argv)
{
	// t10k-labels.idx1-ubyte
	// train-labels.idx1-ubyte
	// t10k-images.idx3-ubyte
	// train-images.idx3-ubyte

	// load in the MNIST examples...
	std::vector<char> traingingLabels = neurosys::MNIST::HandwritingLabelsRead("t10k-labels.idx1-ubyte");
	std::vector<neurosys::layer> traingingImages = neurosys::MNIST::HandwritingImagesRead("t10k-images.idx3-ubyte");


	return 0;

}
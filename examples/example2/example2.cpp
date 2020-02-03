#include "..\include\neurosys.hpp"

int main(int argc, char** argv)
{

	neurosys::network net({
		neurosys::layer(3, neurosys::activation::linear),
		neurosys::layer(2, neurosys::activation::sigmoid, 0.5),
		neurosys::layer(2, neurosys::activation::sigmoid, 0.5)
		});

	// we need to set the weights...
	/*net.weighting(0, 0, { 0.1, 0.2 });
	net.weighting(0, 1, { 0.3, 0.4 });
	net.weighting(0, 2, { 0.5, 0.6 });
	net.weighting(1, 0, { 0.7, 0.8 });
	net.weighting(1, 1, { 0.9, 0.1 });
*/


	return 0;
}
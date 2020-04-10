#include "../../include/neurosys.hpp"

int main(int argc, char** argv)
{
    neurosys::network net(
		neurosys::input(2),
		{ neurosys::layer(2,neurosys::activation::sigmoid, 1.0) },
		neurosys::output(1, neurosys::activation::sigmoid, 1.0)
	);

	std::vector<neurosys::input> inputs({ 
		neurosys::input(std::vector<double>({0 ,0})), 
		neurosys::input(std::vector<double>({0, 1})), 
		neurosys::input(std::vector<double>({1, 0})),
		neurosys::input(std::vector<double>({1, 1})) 
	});

	std::vector<neurosys::output> expected({ 
		neurosys::output(std::vector<double>({0})),
		neurosys::output(std::vector<double>({1.0})),
		neurosys::output(std::vector<double>({1.0})),
		neurosys::output(std::vector<double>({0}))
	});	

	net.reset();

	// epochs 10000, learning rate 0.1
	for (unsigned int n = 0; n < 1000; ++n)
	{
		net = neurosys::train(net, inputs, expected, neurosys::loss::squaredError, 0.1);
		std::cout << '\n';
	}
}
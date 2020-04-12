#include "../../include/neurosys.hpp"
#include <iostream>

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
		neurosys::output(std::vector<double>({1})),
		neurosys::output(std::vector<double>({1})),
		neurosys::output(std::vector<double>({0}))
	});	

	net.reset();

	// Set bias to 1...
	net[1].bias(1.0);
	net[2].bias(1.0);
	
	double cost = 0;
	for (unsigned int n = 0; n < 100000; ++n)
	{
		net = neurosys::epoch(net, inputs, expected, neurosys::loss::function::squaredError, 0.01, [&cost](double C){ cost = C;});
		std::cout << "epoch " << n + 1 << " cost = " << cost << '\n';
	}

	return 0;
}
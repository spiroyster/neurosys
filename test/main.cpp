#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "..\include\neurosys.hpp"

TEST_CASE("modelConstruction11", "[modelConstruction11]")
{
	// single model with single layer...
	neurosys::model m({ 
			neurosys::layer(neurosys::activation::sigmoid, { 42.0 }),
			neurosys::layer(neurosys::activation::sigmoid, { 54.0 }),
		});

	CHECK(m.layerCount() == 3);		// two original layers + synapse layer between them.
	CHECK(m.paramterCount() == 3);
	CHECK(m.input() == m[0]);		// input is first layer...
	CHECK(m.output() == m[1]);		// output is second layer...

	// input layer...
	CHECK(m[0].size() == 1);
	//CHECK(m[0].activationFn() == neurosys::activation::sigmoid);
	CHECK(m[0][0] == 42.0);

	// output layer...
	CHECK(m[1].size() == 1);
	//CHECK(m[1].activationFn() == neurosys::activation::linear);
	CHECK(m[1][0] == 54.0);

	CHECK(m.synapses(0, 0).values().size() == m.output().values().size());
	
}

TEST_CASE("modelConstruction12", "[modelConstruction12]")
{
	neurosys::layer input(neurosys::activation::sigmoid, { 42.0 });
	neurosys::layer output(neurosys::activation::sigmoid, 2, [](std::size_t n) { return static_cast<neurosys::cell>(n); });

	// single model with no hidden layers
	neurosys::model m({ input, output });

	CHECK(m.layerCount() == 3);		// two original layers + synapse layer between them.
	CHECK(m.paramterCount() == 5);
	CHECK(m.input() == input);			// input is first layer...
	CHECK(m.output() == output);		// output is second layer...
	CHECK(m[0] == input);
	CHECK(m[1] == output);
	CHECK(m.synapses(0, 0).size() == output.size());
}

TEST_CASE("modelConstruction22", "[modelConstruction22]")
{
	neurosys::layer input(neurosys::activation::sigmoid, 2, [](std::size_t n) { return static_cast<neurosys::cell>(n); });
	neurosys::layer output(neurosys::activation::sigmoid, 2, [](std::size_t n) { return static_cast<neurosys::cell>(n); });

	// single model with no hidden layers
	neurosys::model m({ input, output });

	CHECK(m.layerCount() == 4);		// two original layers + synapse layer between them.
	CHECK(m.paramterCount() == 8);
	CHECK(m.input() == input);			// input is first layer...
	CHECK(m.output() == output);		// output is second layer...
	CHECK(m[0] == input);
	CHECK(m[1] == output);
	CHECK(m.synapses(0, 0).size() == output.size());
	CHECK(m.synapses(0, 1).size() == output.size());
}

TEST_CASE("modelConstruction33", "[modelConstruction33]")
{
	neurosys::layer input(neurosys::activation::sigmoid, 3, [](std::size_t n) { return static_cast<neurosys::cell>(n); });
	neurosys::layer output(neurosys::activation::sigmoid, 3, [](std::size_t n) { return static_cast<neurosys::cell>(n); });

	// single model with no hidden layers
	neurosys::model m({ input, output });

	CHECK(m.layerCount() == 5);		// two original layers + synapse layer between them.
	CHECK(m.paramterCount() == 15);
	CHECK(m.input() == input);			// input is first layer...
	CHECK(m.output() == output);		// output is second layer...
	CHECK(m[0] == input);
	CHECK(m[1] == output);
	CHECK(m.synapses(0, 0).size() == output.size());
	CHECK(m.synapses(0, 1).size() == output.size());
	CHECK(m.synapses(0, 2).size() == output.size());
}

TEST_CASE("modelConstruction123", "[modelConstruction123]")
{
	neurosys::layer input(neurosys::activation::sigmoid, 1, [](std::size_t n) { return static_cast<neurosys::cell>(n); });
	neurosys::layer hl(neurosys::activation::sigmoid, 2, [](std::size_t n) { return static_cast<neurosys::cell>(n); });
	neurosys::layer output(neurosys::activation::sigmoid, 3, [](std::size_t n) { return static_cast<neurosys::cell>(n); });

	// single model with no hidden layers
	neurosys::model m({ input, hl, output });

	CHECK(m.layerCount() == 6);		// two original layers + synapse layer between them.
	CHECK(m.paramterCount() == 14);
	CHECK(m.input() == input);			// input is first layer...
	CHECK(m.output() == output);		// output is second layer...
	CHECK(m[0] == input);
	CHECK(m[1] == hl);
	CHECK(m[2] == output);

	CHECK(m.synapses(0, 0).size() == hl.size());
	CHECK(m.synapses(1, 0).size() == output.size());
	CHECK(m.synapses(1, 1).size() == output.size());
}

TEST_CASE("iterator123", "[iterator123]")
{
	neurosys::layer input(neurosys::activation::sigmoid, 1, [](std::size_t n) { return static_cast<neurosys::cell>(n); });
	neurosys::layer hl(neurosys::activation::sigmoid, 2, [](std::size_t n) { return static_cast<neurosys::cell>(n); });
	neurosys::layer output(neurosys::activation::sigmoid, 3, [](std::size_t n) { return static_cast<neurosys::cell>(n); });

	// single model with no hidden layers
	neurosys::model m({ input, hl, output });

	neurosys::model::iterator itr = m.cell(0, 0);

	// check not valid to go back...
	CHECK(itr.back(0) == neurosys::model::iterator());
	CHECK(!*itr.back(0));
	
	// check not valid to next or prev...
	CHECK(!*itr.down());
	CHECK(!*itr.up());

	// move to the next...
	itr = itr.forward(0);
	CHECK(*itr);

	// can't go up!
	CHECK(!*itr.up());

	// can go down... once...
	CHECK(*itr.down());
	itr = itr.down();
	CHECK(!*itr.down());

	// go to the next...




}

TEST_CASE("backSynapses123", "[backSynapses123]")
{
	neurosys::layer input(neurosys::activation::sigmoid, 1, [](std::size_t n) { return static_cast<neurosys::cell>(n); });
	neurosys::layer hl(neurosys::activation::sigmoid, 2, [](std::size_t n) { return static_cast<neurosys::cell>(n); });
	neurosys::layer output(neurosys::activation::sigmoid, 3, [](std::size_t n) { return static_cast<neurosys::cell>(n); });

	// single model with no hidden layers
	neurosys::model m({ input, hl, output });

	neurosys::model::iterator itr = m.cell(2, 0);

	// check valid to go back...
	CHECK(*itr.back(0));

	std::vector<neurosys::cell*> bs = itr.backSynapses();

	CHECK(bs.size() == hl.size());

	// change these weigths and see if the model correctly updates...
	*bs[0] = 12.0;
	*bs[1] = 13.0;

	CHECK(m.synapses(1, 0)[0] == 12.0);
	CHECK(m.synapses(1, 1)[0] == 13.0);

}
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "..\include\neurosys.hpp"

namespace
{
}

TEST_CASE("modelConstruction11", "[modelConstruction11]")
{
	// single model with single layer...
	neurosys::model m({ 
			neurosys::layer(1, neurosys::cell(42.0, neurosys::activation::sigmoid)),
			neurosys::layer(1, neurosys::cell(54.0, neurosys::activation::linear)),
		});

	CHECK(m.layerCount() == 3);		// two original layers + synapse layer between them.
	CHECK(m.paramterCount() == 3);
	CHECK(m.input() == m[0]);		// input is first layer...
	CHECK(m.output() == m[1]);		// output is second layer...

	// input layer...
	CHECK(m[0].size() == 1);
	CHECK(m[0][0].second == neurosys::activation::sigmoid);
	CHECK(m[0][0].first == 42.0);

	// output layer...
	CHECK(m[1].size() == 1);
	CHECK(m[1][0].second == neurosys::activation::linear);
	CHECK(m[1][0].first == 54.0);

	CHECK(m.synapses(0, 0).size() == m.output().size());
	
}

TEST_CASE("modelConstruction12", "[modelConstruction12]")
{
	neurosys::layer input(1, neurosys::cell(42.0, neurosys::activation::sigmoid));
	
	neurosys::layer output(2, neurosys::cell(1.0, neurosys::activation::linear));
	{
		unsigned int n = 0;
		std::for_each(output.begin(), output.end(), [&n](neurosys::cell& c) { c.first = static_cast<neurosys::payload>(n); ++n; });
	}
	
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
	neurosys::layer input(2, neurosys::cell(0, neurosys::activation::sigmoid));
	{
		unsigned int n = 0;
		std::for_each(input.begin(), input.end(), [&n](neurosys::cell& c) { c.first = static_cast<neurosys::payload>(n); ++n; });
	}
	
	neurosys::layer output(2, neurosys::cell(0, neurosys::activation::linear));
	{
		unsigned int n = 0;
		std::for_each(output.begin(), output.end(), [&n](neurosys::cell& c) { c.first = static_cast<neurosys::payload>(n); ++n; });
	}
	
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
	neurosys::layer input(3, neurosys::cell(0, neurosys::activation::sigmoid));
	{
		unsigned int n = 0;
		std::for_each(input.begin(), input.end(), [&n](neurosys::cell& c) { c.first = static_cast<neurosys::payload>(n); ++n; });
	}

	neurosys::layer output(3, neurosys::cell(0, neurosys::activation::linear));
	{
		unsigned int n = 0;
		std::for_each(output.begin(), output.end(), [&n](neurosys::cell& c) { c.first = static_cast<neurosys::payload>(n); ++n; });
	}

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
	neurosys::layer input(1, neurosys::cell(0, neurosys::activation::sigmoid));
	{
		unsigned int n = 0;
		std::for_each(input.begin(), input.end(), [&n](neurosys::cell& c) { c.first = static_cast<neurosys::payload>(n); ++n; });
	}

	neurosys::layer hl(2, neurosys::cell(0, neurosys::activation::sigmoid));
	{
		unsigned int n = 0;
		std::for_each(hl.begin(), hl.end(), [&n](neurosys::cell& c) { c.first = static_cast<neurosys::payload>(n); ++n; });
	}

	neurosys::layer output(3, neurosys::cell(0, neurosys::activation::linear));
	{
		unsigned int n = 0;
		std::for_each(output.begin(), output.end(), [&n](neurosys::cell& c) { c.first = static_cast<neurosys::payload>(n); ++n; });
	}

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
	neurosys::layer input(1, neurosys::cell(0, neurosys::activation::sigmoid));
	{
		unsigned int n = 0;
		std::for_each(input.begin(), input.end(), [&n](neurosys::cell& c) { c.first = static_cast<neurosys::payload>(n); ++n; });
	}

	neurosys::layer hl(2, neurosys::cell(0, neurosys::activation::sigmoid));
	{
		unsigned int n = 0;
		std::for_each(hl.begin(), hl.end(), [&n](neurosys::cell& c) { c.first = static_cast<neurosys::payload>(n); ++n; });
	}

	neurosys::layer output(3, neurosys::cell(0, neurosys::activation::linear));
	{
		unsigned int n = 0;
		std::for_each(output.begin(), output.end(), [&n](neurosys::cell& c) { c.first = static_cast<neurosys::payload>(n); ++n; });
	}

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
	neurosys::layer input(1, neurosys::cell(0, neurosys::activation::sigmoid));
	{
		unsigned int n = 0;
		std::for_each(input.begin(), input.end(), [&n](neurosys::cell& c) { c.first = static_cast<neurosys::payload>(n); ++n; });
	}

	neurosys::layer hl(2, neurosys::cell(0, neurosys::activation::sigmoid));
	{
		unsigned int n = 0;
		std::for_each(hl.begin(), hl.end(), [&n](neurosys::cell& c) { c.first = static_cast<neurosys::payload>(n); ++n; });
	}

	neurosys::layer output(3, neurosys::cell(0, neurosys::activation::linear));
	{
		unsigned int n = 0;
		std::for_each(output.begin(), output.end(), [&n](neurosys::cell& c) { c.first = static_cast<neurosys::payload>(n); ++n; });
	}

	// single model with no hidden layers
	neurosys::model m({ input, hl, output });

	neurosys::model::iterator itr = m.cell(2, 0);

	// check valid to go back...
	CHECK(*itr.back(0));

	std::vector<neurosys::cell*> bs = itr.backSynapses();

	CHECK(bs.size() == hl.size());

	// cheange these weigths and see if the model correctly updates...
	bs[0]->first = 12.0;
	bs[1]->first = 13.0;

	CHECK(m.synapses(1, 0)[0].first == 12.0);
	CHECK(m.synapses(1, 1)[0].first == 13.0);

}
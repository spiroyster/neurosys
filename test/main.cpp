#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "../include/neurosys.hpp"

namespace
{
    bool ApproxEquals(const double& a, const double& b)
    {
        return sqrt((b-a)*(b-a)) < 0.01;
    }
}

TEST_CASE("activation linear", "[activation]")
{
	CHECK(neurosys::activation::Fn[neurosys::activation::function::linear](0) == Approx(0));
	CHECK(neurosys::activation::Fn[neurosys::activation::function::linear](0.1) == Approx(0.1));
	CHECK(neurosys::activation::Fn[neurosys::activation::function::linear](0.2) == Approx(0.2));
	CHECK(neurosys::activation::Fn[neurosys::activation::function::linear](0.3) == Approx(0.3));
	CHECK(neurosys::activation::Fn[neurosys::activation::function::linear](0.4) == Approx(0.4));
	CHECK(neurosys::activation::Fn[neurosys::activation::function::linear](0.5) == Approx(0.5));
	CHECK(neurosys::activation::Fn[neurosys::activation::function::linear](0.6) == Approx(0.6));
	CHECK(neurosys::activation::Fn[neurosys::activation::function::linear](0.7) == Approx(0.7));
	CHECK(neurosys::activation::Fn[neurosys::activation::function::linear](0.8) == Approx(0.8));
	CHECK(neurosys::activation::Fn[neurosys::activation::function::linear](0.9) == Approx(0.9));
	CHECK(neurosys::activation::Fn[neurosys::activation::function::linear](1.0) == Approx(1.0));
}
TEST_CASE("activation sigmoid", "[activation]")
{
	CHECK(neurosys::activation::Fn[neurosys::activation::function::sigmoid](0) == Approx(0.5));
	CHECK(neurosys::activation::Fn[neurosys::activation::function::sigmoid](0.1) == Approx(0.5249791875));
	CHECK(neurosys::activation::Fn[neurosys::activation::function::sigmoid](0.2) == Approx(0.5498339973));
	CHECK(neurosys::activation::Fn[neurosys::activation::function::sigmoid](0.3) == Approx(0.5744425168));
	CHECK(neurosys::activation::Fn[neurosys::activation::function::sigmoid](0.4) == Approx(0.5986876601));
	CHECK(neurosys::activation::Fn[neurosys::activation::function::sigmoid](0.5) == Approx(0.6224593312));
	CHECK(neurosys::activation::Fn[neurosys::activation::function::sigmoid](0.6) == Approx(0.6456563062));
	CHECK(neurosys::activation::Fn[neurosys::activation::function::sigmoid](0.7) == Approx(0.6681877722));
	CHECK(neurosys::activation::Fn[neurosys::activation::function::sigmoid](0.8) == Approx(0.6899744811));
	CHECK(neurosys::activation::Fn[neurosys::activation::function::sigmoid](0.9) == Approx(0.7109495026));
	CHECK(neurosys::activation::Fn[neurosys::activation::function::sigmoid](1.0) == Approx(0.7310585786));
}
TEST_CASE("activation linearPrime", "[activation][derivative]")
{
	CHECK(neurosys::activation::FnPrime[neurosys::activation::function::linear](0) == Approx(1.0));
	CHECK(neurosys::activation::FnPrime[neurosys::activation::function::linear](0.1) == Approx(1.0));
	CHECK(neurosys::activation::FnPrime[neurosys::activation::function::linear](0.2) == Approx(1.0));
	CHECK(neurosys::activation::FnPrime[neurosys::activation::function::linear](0.3) == Approx(1.0));
	CHECK(neurosys::activation::FnPrime[neurosys::activation::function::linear](0.4) == Approx(1.0));
	CHECK(neurosys::activation::FnPrime[neurosys::activation::function::linear](0.5) == Approx(1.0));
	CHECK(neurosys::activation::FnPrime[neurosys::activation::function::linear](0.6) == Approx(1.0));
	CHECK(neurosys::activation::FnPrime[neurosys::activation::function::linear](0.7) == Approx(1.0));
	CHECK(neurosys::activation::FnPrime[neurosys::activation::function::linear](0.8) == Approx(1.0));
	CHECK(neurosys::activation::FnPrime[neurosys::activation::function::linear](0.9) == Approx(1.0));
	CHECK(neurosys::activation::FnPrime[neurosys::activation::function::linear](1.0) == Approx(1.0));
	
}
TEST_CASE("activation sigmoidPrime", "[activation][derivative]")
{
	CHECK(neurosys::activation::FnPrime[neurosys::activation::function::sigmoid](0) == Approx(0));
	CHECK(neurosys::activation::FnPrime[neurosys::activation::function::sigmoid](0.1) == Approx(0.09));
	CHECK(neurosys::activation::FnPrime[neurosys::activation::function::sigmoid](0.2) == Approx(0.16));
	CHECK(neurosys::activation::FnPrime[neurosys::activation::function::sigmoid](0.3) == Approx(0.21));
	CHECK(neurosys::activation::FnPrime[neurosys::activation::function::sigmoid](0.4) == Approx(0.24));
	CHECK(neurosys::activation::FnPrime[neurosys::activation::function::sigmoid](0.5) == Approx(0.25));
	CHECK(neurosys::activation::FnPrime[neurosys::activation::function::sigmoid](0.6) == Approx(0.24));
	CHECK(neurosys::activation::FnPrime[neurosys::activation::function::sigmoid](0.7) == Approx(0.21));
	CHECK(neurosys::activation::FnPrime[neurosys::activation::function::sigmoid](0.8) == Approx(0.16));
	CHECK(neurosys::activation::FnPrime[neurosys::activation::function::sigmoid](0.9) == Approx(0.09));
	CHECK(neurosys::activation::FnPrime[neurosys::activation::function::sigmoid](1.0) == Approx(0));
}

// squared error... cross entropy...


// mean, median, sum etc...


TEST_CASE("matrix empty", "[matrix]")
{
	{
		neurosys::matrix m(0, 0);
		CHECK(m.size() == 0);
		CHECK(!m.isColumnVector());
		CHECK(!m.isRowVector());

	}
	{
		neurosys::matrix m(1, 0);
		CHECK(m.size() == 0);
		CHECK(!m.isColumnVector());
		CHECK(!m.isRowVector());
	}
	{
		neurosys::matrix m(0, 1);
		CHECK(m.size() == 0);
		CHECK(!m.isColumnVector());
		CHECK(!m.isRowVector());
	}
}

TEST_CASE("matrix square construction", "[matrix][construct]")
{
	{
		neurosys::matrix m(1, 1);
		CHECK(m.size() == 1);
		CHECK(m.isColumnVector());
		CHECK(m.isRowVector());
		CHECK(m.value(0, 0) == 0);

	}
	{
		neurosys::matrix m({{ 42.0 }}, 1);
		CHECK(m.size() == 1);
		CHECK(m.isColumnVector());
		CHECK(m.isRowVector());
		CHECK(m.value(0, 0) == 42.0);
	}
	{
		neurosys::matrix m(2, 2);
		CHECK(m.size() == 4);
		CHECK(!m.isColumnVector());
		CHECK(!m.isRowVector());
		CHECK(m.value(0, 0) == 0);
		CHECK(m.value(0, 1) == 0);
		CHECK(m.value(1, 0) == 0);
		CHECK(m.value(1, 1) == 0);
	}
	{
		neurosys::matrix m({ 1.0, 2.0, 3.0, 4.0 }, 2);
		CHECK(m.size() == 4);
		CHECK(!m.isColumnVector());
		CHECK(!m.isRowVector());
		CHECK(m.value(0, 0) == 1.0);
		CHECK(m.value(0, 1) == 2.0);
		CHECK(m.value(1, 0) == 3.0);
		CHECK(m.value(1, 1) == 4.0);
	}
	
}

TEST_CASE("matrix mn construct", "[matrix][construct]")
{
	{
		neurosys::matrix m(1, 2);
		CHECK(m.size() == 2);
		CHECK(!m.isColumnVector());
		CHECK(m.isRowVector());
		CHECK(m.value(0, 0) == 0);
		CHECK(m.value(0, 1) == 0);
	}
	{
		neurosys::matrix m({ 1.0, 2.0 }, 1);
		CHECK(m.size() == 2);
		CHECK(m.isColumnVector());
		CHECK(!m.isRowVector());
		CHECK(m.value(0, 0) == 1.0);
		CHECK(m.value(0, 1) == 2.0);
	}
	{
		neurosys::matrix m({ 1.0, 2.0 }, 1);
		CHECK(m.size() == 2);
		CHECK(m.isColumnVector());
		CHECK(!m.isRowVector());
		CHECK(m.value(0, 0) == 1.0);
		CHECK(m.value(1, 0) == 2.0);
	}
	{
		neurosys::matrix m({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 }, 3);
		CHECK(m.size() == 12);
		CHECK(!m.isColumnVector());
		CHECK(!m.isRowVector());
		CHECK(m.value(0, 0) == 1.0);
		CHECK(m.value(0, 1) == 2.0);
		CHECK(m.value(0, 2) == 3.0);
		CHECK(m.value(1, 0) == 4.0);
		CHECK(m.value(1, 1) == 5.0);
		CHECK(m.value(1, 2) == 6.0);
		CHECK(m.value(2, 0) == 7.0);
		CHECK(m.value(2, 1) == 8.0);
		CHECK(m.value(2, 2) == 9.0);
		CHECK(m.value(3, 0) == 10.0);
		CHECK(m.value(3, 1) == 11.0);
		CHECK(m.value(3, 2) == 12.0);
	}
	{
		neurosys::matrix m(4, 3);
		CHECK(m.size() == 12);
		CHECK(!m.isColumnVector());
		CHECK(!m.isRowVector());
		CHECK(m.value(0, 0) == 0);
		CHECK(m.value(0, 1) == 0);
		CHECK(m.value(0, 2) == 0);
		CHECK(m.value(1, 0) == 0);
		CHECK(m.value(1, 1) == 0);
		CHECK(m.value(1, 2) == 0);
		CHECK(m.value(2, 0) == 0);
		CHECK(m.value(2, 1) == 0);
		CHECK(m.value(2, 2) == 0);
		CHECK(m.value(3, 0) == 0);
		CHECK(m.value(3, 1) == 0);
		CHECK(m.value(3, 2) == 0);
	}
	{
		neurosys::matrix m(3, 1);
		CHECK(m.size() == 3);
		CHECK(m.isColumnVector());
		CHECK(!m.isRowVector());
		CHECK(m.value(0, 0) == 0);
		CHECK(m.value(1, 0) == 0);
		CHECK(m.value(2, 0) == 0);
	}
	{
		neurosys::matrix m({ 1.0, 2.0, 3.0 }, 1);
		CHECK(m.size() == 3);
		CHECK(m.isColumnVector());
		CHECK(!m.isRowVector());
		CHECK(m.value(0, 0) == 1.0);
		CHECK(m.value(1, 0) == 2.0);
		CHECK(m.value(2, 0) == 3.0);
	}
}

TEST_CASE("neurons construct", "[neurons][construct]")
{
	{
		neurosys::neurons m(0);
		CHECK(m.size() == 0);
		CHECK(!m.isColumnVector());
		CHECK(!m.isRowVector());
	}
	{
		neurosys::neurons m(1);
		CHECK(m.size() == 1);
		CHECK(m.isColumnVector());
		CHECK(m.isRowVector());
		CHECK(m.value(0) == 0);
	}
	{
		neurosys::neurons m(std::vector<double>({ 42.0 }));
		CHECK(m.size() == 1);
		CHECK(m.isColumnVector());
		CHECK(m.isRowVector());
		CHECK(m.value(0) == 42.0);
	}
	{
		neurosys::neurons m(2);
		CHECK(m.size() == 2);
		CHECK(m.isColumnVector());
		CHECK(!m.isRowVector());
		CHECK(m.value(0) == 0);
		CHECK(m.value(1) == 0);
	}
	{
		neurosys::neurons m(std::vector<double>({ 1.0, 2.0 }));
		CHECK(m.size() == 2);
		CHECK(m.isColumnVector());
		CHECK(!m.isRowVector());
		CHECK(m.value(0) == 1.0);
		CHECK(m.value(1) == 2.0);
	}
	{
		neurosys::neurons m(3);
		CHECK(m.size() == 3);
		CHECK(m.isColumnVector());
		CHECK(!m.isRowVector());
		CHECK(m.value(0) == 0);
		CHECK(m.value(1) == 0);
		CHECK(m.value(2) == 0);
	}
	{
		neurosys::neurons m({ 1.0, 2.0, 3.0 });
		CHECK(m.size() == 3);
		CHECK(m.isColumnVector());
		CHECK(!m.isRowVector());
		CHECK(m.value(0) == 1.0);
		CHECK(m.value(1) == 2.0);
		CHECK(m.value(2) == 3.0);
	}
	{
		neurosys::neurons m(10);
		CHECK(m.size() == 10);
		CHECK(m.isColumnVector());
		CHECK(!m.isRowVector());
		CHECK(m.value(0) == 0);
		CHECK(m.value(1) == 0);
		CHECK(m.value(2) == 0);
		CHECK(m.value(3) == 0);
		CHECK(m.value(4) == 0);
		CHECK(m.value(5) == 0);
		CHECK(m.value(6) == 0);
		CHECK(m.value(7) == 0);
		CHECK(m.value(8) == 0);
		CHECK(m.value(9) == 0);
	}
	{
		neurosys::neurons m({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 });
		CHECK(m.size() == 10);
		CHECK(m.isColumnVector());
		CHECK(!m.isRowVector());
		CHECK(m.value(0) == 1.0);
		CHECK(m.value(1) == 2.0);
		CHECK(m.value(2) == 3.0);
		CHECK(m.value(3) == 4.0);
		CHECK(m.value(4) == 5.0);
		CHECK(m.value(5) == 6.0);
		CHECK(m.value(6) == 7.0);
		CHECK(m.value(7) == 8.0);
		CHECK(m.value(8) == 9.0);
		CHECK(m.value(9) == 10.0);
	}
}

TEST_CASE("matrix square transpose", "[matrix][transpose]")
{
	{
		neurosys::matrix m({ { 1.0 } }, 1);
		neurosys::matrix n = neurosys::maths::transpose(m);

		CHECK(n.value(0, 0) == 1.0);
	}
	{
		neurosys::matrix m({ 1.0, 2.0, 3.0, 4.0 }, 2);
		neurosys::matrix n = neurosys::maths::transpose(m);

		CHECK(n.value(0, 0) == 1.0);
		CHECK(n.value(0, 1) == 3.0);
		CHECK(n.value(1, 0) == 2.0);
		CHECK(n.value(1, 1) == 4.0);
	}
	{
		neurosys::matrix m({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 }, 3);
		neurosys::matrix n = neurosys::maths::transpose(m);

		CHECK(n.value(0, 0) == 1.0);
		CHECK(n.value(0, 1) == 4.0);
		CHECK(n.value(0, 2) == 7.0);
		CHECK(n.value(1, 0) == 2.0);
		CHECK(n.value(1, 1) == 5.0);
		CHECK(n.value(1, 2) == 8.0);
		CHECK(n.value(2, 0) == 3.0);
		CHECK(n.value(2, 1) == 6.0);
		CHECK(n.value(2, 2) == 9.0);
	}
}
TEST_CASE("matrix mn transpose", "[matrix][transpose]")
{
	{
		neurosys::matrix m({ { 1.0, 2.0 } }, 2);
		neurosys::matrix n = neurosys::maths::transpose(m);

		CHECK(n.m() == 2);
		CHECK(n.n() == 1);
		CHECK(n.value(0, 0) == 1.0);
		CHECK(n.value(1, 0) == 2.0);
	}
	{
		neurosys::matrix m({ { 1.0, 2.0 } }, 1);
		neurosys::matrix n = neurosys::maths::transpose(m);
		CHECK(n.m() == 1);
		CHECK(n.n() == 2);
		CHECK(n.value(0, 0) == 1.0);
		CHECK(n.value(0, 1) == 2.0);
	}
	{
		neurosys::matrix m({ { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 } }, 3);
		neurosys::matrix n = neurosys::maths::transpose(m);
		CHECK(n.m() == 3);
		CHECK(n.n() == 2);
		CHECK(n.value(0, 0) == 1.0);
		CHECK(n.value(0, 1) == 4.0);
		CHECK(n.value(1, 0) == 2.0);
		CHECK(n.value(1, 1) == 5.0);
		CHECK(n.value(2, 0) == 3.0);
		CHECK(n.value(2, 1) == 6.0);
	}
	{
		neurosys::matrix m({ { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 } }, 2);
		neurosys::matrix n = neurosys::maths::transpose(m);
		CHECK(n.m() == 2);
		CHECK(n.n() == 3);
		CHECK(n.value(0, 0) == 1.0);
		CHECK(n.value(0, 1) == 3.0);
		CHECK(n.value(0, 2) == 5.0);
		CHECK(n.value(1, 0) == 2.0);
		CHECK(n.value(1, 1) == 4.0);
		CHECK(n.value(1, 2) == 6.0);
	}
}

TEST_CASE("matrix multiply", "[matrix][multiply]")
{
	{
		neurosys::matrix m({ {1.0} }, 1);
		neurosys::matrix n({ {1.0} }, 1);
		neurosys::matrix result = neurosys::maths::product(m, n);

		CHECK(result.size() == 1);
		CHECK(result.value(0, 0) == 1.0);
	}
	{
		neurosys::matrix m({ {1.0, 2.0, 3.0 } }, 3);
		neurosys::matrix n({ {1.0, 2.0, 3.0 } }, 1);
		neurosys::matrix result = neurosys::maths::product(m, n);

		CHECK(result.size() == 1);
		CHECK(result.value(0, 0) == 14.0);
	}
	{
		neurosys::matrix m({ {1.0, 2.0, 3.0 } }, 1);
		neurosys::matrix n({ {1.0, 2.0, 3.0 } }, 3);
		neurosys::matrix result = neurosys::maths::product(m, n);

		CHECK(result.size() == 9);
		CHECK(result.value(0, 0) == 1.0);
        CHECK(result.value(0, 1) == 2.0);
        CHECK(result.value(0, 2) == 3.0);
        CHECK(result.value(1, 0) == 2.0);
        CHECK(result.value(1, 1) == 4.0);
        CHECK(result.value(1, 2) == 6.0);
        CHECK(result.value(2, 0) == 3.0);
        CHECK(result.value(2, 1) == 6.0);
        CHECK(result.value(2, 2) == 9.0);
	}
	{
		neurosys::matrix m({ {1.0, 2.0, 3.0, 4.0 } }, 2);
		neurosys::matrix n({ {1.0, 2.0, 3.0, 4.0 } }, 2);
		neurosys::matrix result = neurosys::maths::product(m, n);

		CHECK(result.size() == 4);
		CHECK(result.value(0, 0) == 7.0);
		CHECK(result.value(0, 1) == 10.0);
		CHECK(result.value(1, 0) == 15.0);
		CHECK(result.value(1, 1) == 22.0);
	}
	
}

TEST_CASE("matrix add scalar", "[matrix][add]")
{
	{
		neurosys::matrix m({ {1.0} }, 1);
		neurosys::matrix result = neurosys::maths::add(m, 1.0);
		CHECK(result.size() == 1);
		CHECK(result.value(0, 0) == 2.0);
	}
	{
		neurosys::matrix m({1.0, 2.0, 3.0}, 1);
		neurosys::matrix result = neurosys::maths::add(m, 1.0);
		CHECK(result.size() == 3);
		CHECK(result.value(0, 0) == 2.0);
		CHECK(result.value(1, 0) == 3.0);
		CHECK(result.value(2, 0) == 4.0);
	}
	{
		neurosys::matrix m({ 1.0, 2.0, 3.0 }, 3);
		neurosys::matrix result = neurosys::maths::add(m, 1.0);
		CHECK(result.size() == 3);
		CHECK(result.value(0, 0) == 2.0);
		CHECK(result.value(0, 1) == 3.0);
		CHECK(result.value(0, 2) == 4.0);
	}
	{
		neurosys::matrix m({ 1.0, 2.0, 3.0, 4.0 }, 2);
		neurosys::matrix result = neurosys::maths::add(m, 1.0);
		CHECK(result.size() == 4);
		CHECK(result.value(0, 0) == 2.0);
		CHECK(result.value(0, 1) == 3.0);
		CHECK(result.value(1, 0) == 4.0);
		CHECK(result.value(1, 1) == 5.0);
	}
	{
		neurosys::matrix m({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, 3);
		neurosys::matrix result = neurosys::maths::add(m, 1.0);
		CHECK(result.size() == 6);
		CHECK(result.value(0, 0) == 2.0);
		CHECK(result.value(0, 1) == 3.0);
		CHECK(result.value(0, 2) == 4.0);
		CHECK(result.value(1, 0) == 5.0);
		CHECK(result.value(1, 1) == 6.0);
		CHECK(result.value(1, 2) == 7.0);
	}
	{
		neurosys::matrix m({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, 2);
		neurosys::matrix result = neurosys::maths::add(m, 1.0);
		CHECK(result.size() == 6);
		CHECK(result.value(0, 0) == 2.0);
		CHECK(result.value(0, 1) == 3.0);
		CHECK(result.value(1, 0) == 4.0);
		CHECK(result.value(1, 1) == 5.0);
		CHECK(result.value(2, 0) == 6.0);
		CHECK(result.value(2, 1) == 7.0);
	}

}

TEST_CASE("matrix add matrix", "[matrix][add]")
{
	{
		neurosys::matrix m({ {1.0} }, 1);
		neurosys::matrix n({ {1.0} }, 1);
		neurosys::matrix result = neurosys::maths::add(m, n);
		CHECK(result.size() == 1);
		CHECK(result.value(0, 0) == 2.0);
	}
	
}

// layer...
//TEST_CASE("layer construct", "[layer]")
//{
//}

TEST_CASE("perceptron construct", "[perceptron][network][construct]")
{
    {
        neurosys::network net(neurosys::input(1), {}, neurosys::output(1, neurosys::activation::linear, 1.0));
        CHECK(net.size() == 2);
        CHECK(net[0].size() == 1);
        CHECK(net[0].weights().size() == 1); 
        CHECK(net[1].size() == 1);
        CHECK(net[1].weights().size() == 1);
    }
    {
        neurosys::network net(neurosys::input(2), {}, neurosys::output(2, neurosys::activation::linear, 1.0));
        CHECK(net.size() == 2);
        CHECK(net[0].size() == 2);
        CHECK(net[0].weights().size() == 2);        
        CHECK(net[1].size() == 2);        
        CHECK(net[1].weights().size() == 4); 
    }
    {
        neurosys::network net(neurosys::input(3), {}, neurosys::output(3, neurosys::activation::linear, 1.0));
        CHECK(net.size() == 2);
        CHECK(net[0].size() == 3);
        CHECK(net[0].weights().size() == 3);        
        CHECK(net[1].size() == 3);        
        CHECK(net[1].weights().size() == 9); 
    }
    {
        neurosys::network net(neurosys::input(3), {}, neurosys::output(1, neurosys::activation::linear, 1.0));
        CHECK(net.size() == 2);
        CHECK(net[0].size() == 3);
        CHECK(net[0].weights().size() == 3);        
        CHECK(net[1].size() == 1);        
        CHECK(net[1].weights().size() == 3);        
    }
    {
        neurosys::network net(neurosys::input(1), {}, neurosys::output(3, neurosys::activation::linear, 1.0));
        CHECK(net.size() == 2);
        CHECK(net[0].size() == 1);
        CHECK(net[0].weights().size() == 1);        
        CHECK(net[1].size() == 3);        
        CHECK(net[1].weights().size() == 3);        
    }
}



TEST_CASE("network construct", "[network][construct]")
{
    
    {
        neurosys::network net(neurosys::input(1), { neurosys::layer(1, neurosys::activation::linear) }, neurosys::output(1, neurosys::activation::linear, 1.0));
        CHECK(net.size() == 3);        
        CHECK(net[0].size() == 1);
        CHECK(net[0].weights().size() == 1);        
        CHECK(net[1].size() == 1);        
        CHECK(net[1].weights().size() == 1);        
        CHECK(net[2].size() == 1);        
        CHECK(net[2].weights().size() == 1);        
    }
    {
        neurosys::network net(neurosys::input(2), { neurosys::layer(2, neurosys::activation::linear) }, neurosys::output(2, neurosys::activation::linear, 1.0));
        CHECK(net.size() == 3);        
        CHECK(net[0].size() == 2);
        CHECK(net[0].weights().size() == 2);        
        CHECK(net[1].size() == 2);        
        CHECK(net[1].weights().size() == 4);        
        CHECK(net[2].size() == 2);        
        CHECK(net[2].weights().size() == 4);        
    }
    {
        neurosys::network net(neurosys::input(3), { neurosys::layer(3, neurosys::activation::linear) }, neurosys::output(3, neurosys::activation::linear, 1.0));
        CHECK(net.size() == 3);        
        CHECK(net[0].size() == 3);
        CHECK(net[0].weights().size() == 3);        
        CHECK(net[1].size() == 3);        
        CHECK(net[1].weights().size() == 9);        
        CHECK(net[2].size() == 3);        
        CHECK(net[2].weights().size() == 9);        
    }
}

TEST_CASE("feedForward z", "[feedForward][z]")
{



}

TEST_CASE("feedForward a", "[feedForward][a]")
{



}

TEST_CASE("feedForward observation", "[feedForward][observation]")
{



}

TEST_CASE("feedForward backPropagate", "[feedForward][backPropagate]")
{
    
}

// train...


TEST_CASE("anotsorandomwalk", "[anotsorandomwalk]")
{
	// https://www.anotsorandomwalk.com/backpropagation-example-with-numbers-step-by-step/

	neurosys::network net(
		neurosys::input(3),
		{ neurosys::layer(2,neurosys::activation::sigmoid, 0.5) },
		neurosys::output(2, neurosys::activation::sigmoid, 0.5)
	);

	net[1].weight(0, 0) = 0.1;
	net[1].weight(0, 1) = 0.2;
	net[1].weight(1, 0) = 0.3;
	net[1].weight(1, 1) = 0.4;
	net[1].weight(2, 0) = 0.5;
	net[1].weight(2, 1) = 0.6;
	net[2].weight(0, 0) = 0.7;
	net[2].weight(0, 1) = 0.8;
	net[2].weight(1, 0) = 0.9;
	net[2].weight(1, 1) = 0.1;

	neurosys::input in({ 1.0, 4.0, 5.0 });

	std::vector<neurosys::neurons> result = neurosys::feedForward::observation(net, in);

    CHECK(result[1].value(0) == Approx(0.9866130822));
	CHECK(result[1].value(1) == Approx(0.9950331983));
    CHECK(result[2].value(0) == Approx(0.889550614));
    CHECK(result[2].value(1) == Approx(0.8004));
    
	// Perform the back prop.
	neurosys::output expected(neurosys::neurons(std::vector<double>({ 0.1, 0.05 })));
	neurosys::network bp = neurosys::feedForward::backPropagate(net, in, expected.neurons(), neurosys::cost::squaredError, 0.01);
	
	// Check the weights of the resultant net.
	CHECK(bp[1].weight(0, 0) == Approx(0.0996679595));
	CHECK(bp[1].weight(0, 1) == Approx(0.1998484889));
	CHECK(bp[1].weight(1, 0) == Approx(0.298671838));
	CHECK(bp[1].weight(1, 1) == Approx(0.3993939555));
	CHECK(bp[1].weight(2, 0) == Approx(0.4983397975));
	CHECK(bp[1].weight(2, 1) == Approx(0.5992424443));
	CHECK(bp[1].bias() == Approx(0.4997582242));

	CHECK(bp[2].weight(0, 0) == Approx(0.6987056418));
	CHECK(bp[2].weight(0, 1) == Approx(0.7986133166));
	CHECK(bp[2].weight(1, 0) == Approx(0.8986945953));
	CHECK(bp[2].weight(1, 1) == Approx(0.0986014821));
	CHECK(bp[2].bias() == Approx(0.4986412902));

}

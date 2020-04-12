#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "../include/neurosys.hpp"

namespace
{
   bool ApproxEquals(const double& a, const double& b)
   {
       return sqrt((b-a)*(b-a)) < 0.01;
   }

	bool ApproxEquals(const neurosys::neurons& a, const neurosys::neurons& b)
	{
		for (unsigned int n = 0; n < a.values().size(); ++n)
			if (a.values()[n] != Approx(b.values()[n]))
				return false;

		return true;
	}
}

/*
 _______  _______  _______  ___   __   __  _______  _______  ___   _______  __    _ 
|   _   ||       ||       ||   | |  | |  ||   _   ||       ||   | |       ||  |  | |
|  |_|  ||       ||_     _||   | |  |_|  ||  |_|  ||_     _||   | |   _   ||   |_| |
|       ||       |  |   |  |   | |       ||       |  |   |  |   | |  | |  ||       |
|       ||      _|  |   |  |   | |       ||       |  |   |  |   | |  |_|  ||  _    |
|   _   ||     |_   |   |  |   |  |     | |   _   |  |   |  |   | |       || | |   |
|__| |__||_______|  |___|  |___|   |___|  |__| |__|  |___|  |___| |_______||_|  |__|
                                                                                                    
*/

TEST_CASE("activation linear", "[activation][linear]")
{
	neurosys::neurons in({ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });
	neurosys::neurons expected({ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });
	neurosys::neurons result = neurosys::activation::Fn[neurosys::activation::function::linear](in);
	for (unsigned int n = 0; n < in.size(); ++n)
		CHECK(result[n] == Approx(expected[n]));
}
TEST_CASE("activation sigmoid", "[activation][sigmoid]")
{
	neurosys::neurons in({ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });
	neurosys::neurons expected({ 0.5, 0.5249791875, 0.5498339973, 0.5744425168, 0.5986876601, 0.6224593312, 0.6456563062, 0.6681877722, 0.6899744811, 0.7109495026, 0.7310585786 });
	neurosys::neurons result = neurosys::activation::Fn[neurosys::activation::function::sigmoid](in);
	for (unsigned int n = 0; n < in.size(); ++n)
		CHECK(result[n] == Approx(expected[n]));
}
TEST_CASE("activation softmax", "[activation][softmax]")
{
	neurosys::neurons in({ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });
	neurosys::neurons expected({ 0.05247615059, 0.05799511552, 0.06409451507, 0.07083539406, 0.07828521748, 0.08651854568, 0.09561778056, 0.1056739903, 0.1167878209, 0.1290705032, 0.1426449666 });
	neurosys::neurons result = neurosys::activation::Fn[neurosys::activation::function::softMax](in);
	for (unsigned int n = 0; n < in.size(); ++n)
		CHECK(result[n] == Approx(expected[n]));
}
TEST_CASE("activation tanh", "[activation][tanh]")
{
	neurosys::neurons in({ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });
	neurosys::neurons expected({ 0, 0.09966799462, 0.1973753202, 0.2913126125, 0.3799489623, 0.4621171573, 0.537049567, 0.6043677771, 0.6640367703, 0.7162978702, 0.761594156 });
	neurosys::neurons result = neurosys::activation::Fn[neurosys::activation::function::tanh](in);
	for (unsigned int n = 0; n < in.size(); ++n)
		CHECK(result[n] == Approx(expected[n]));
}
TEST_CASE("activation linearPrime", "[activation][prime][linear]")
{
	neurosys::neurons in({ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });
	neurosys::neurons expected({ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 });
	neurosys::neurons result = neurosys::activation::FnPrime[neurosys::activation::function::linear](in);
	for (unsigned int n = 0; n < in.size(); ++n)
		CHECK(result[n] == Approx(expected[n]));
}
TEST_CASE("activation sigmoidPrime", "[activation][prime][sigmoid]")
{
	neurosys::neurons in({ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });
	neurosys::neurons expected({ 0, 0.09, 0.16, 0.21, 0.24, 0.25, 0.24, 0.21, 0.16, 0.09, 0 });
	neurosys::neurons result = neurosys::activation::FnPrime[neurosys::activation::function::sigmoid](in);
	for (unsigned int n = 0; n < in.size(); ++n)
		CHECK(result[n] == Approx(expected[n]));
}
TEST_CASE("activation softmaxPrime", "[activation][prime][softmax]")
{
	neurosys::neurons in({ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });
	neurosys::neurons expected({ 0, 0.09, 0.16, 0.21, 0.24, 0.25, 0.24, 0.21, 0.16, 0.09, 0 });
	neurosys::neurons result = neurosys::activation::FnPrime[neurosys::activation::function::softMax](in);
	for (unsigned int n = 0; n < in.size(); ++n)
		CHECK(result[n] == Approx(expected[n]));
}
TEST_CASE("activation tanhPrime", "[activation][prime][tanh]")
{
	neurosys::neurons in({ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });
	neurosys::neurons expected({ 1.0, 0.9900662908, 0.961042983, 0.9151369618, 0.8556387861, 0.786447733, 0.7115777626, 0.63473959, 0.5590551677, 0.4869173611, 0.4199743416 });
	neurosys::neurons result = neurosys::activation::FnPrime[neurosys::activation::function::tanh](in);
	for (unsigned int n = 0; n < in.size(); ++n)
		CHECK(result[n] == Approx(expected[n]));
}

/*
 ___      _______  _______  _______ 
|   |    |       ||       ||       |
|   |    |   _   ||  _____||  _____|
|   |    |  | |  || |_____ | |_____ 
|   |___ |  |_|  ||_____  ||_____  |
|       ||       | _____| | _____| |
|_______||_______||_______||_______|

*/

// squared error...
TEST_CASE("loss squaredError", "[loss][squaredError]")
{
	// ((o - p)^2)/2
	{
		neurosys::neurons observed({ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });
		neurosys::neurons predicted({ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
		
		neurosys::neurons expected({ 0, 0.005, 0.02, 0.045, 0.08, 0.125, 0.18, 0.245, 0.32, 0.405, 0.5 });
		neurosys::neurons result = neurosys::loss::Fn[neurosys::loss::function::squaredError](observed, predicted);
		for (unsigned int n = 0; n < observed.size(); ++n)
			CHECK(result[n] == Approx(expected[n]));
	}
	{
		neurosys::neurons observed({ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });
		neurosys::neurons predicted({ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });
		
		neurosys::neurons expected({ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
		neurosys::neurons result = neurosys::loss::Fn[neurosys::loss::function::squaredError](observed, predicted);
		for (unsigned int n = 0; n < observed.size(); ++n)
			CHECK(result[n] == Approx(expected[n]));
	}
	
}
TEST_CASE("loss squaredError prime", "[loss][squaredError][prime]")
{
	// (o - p)
	neurosys::neurons observed({ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });
	neurosys::neurons predicted({ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });
	
	neurosys::neurons expected({ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
	neurosys::neurons result = neurosys::loss::FnPrime[neurosys::loss::function::squaredError](observed, predicted);
	for (unsigned int n = 0; n < observed.size(); ++n)
		CHECK(result[n] == Approx(expected[n]));
}
TEST_CASE("cost squaredError", "[cost][squaredError]")
{
	{
		neurosys::neurons observed({ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });
		neurosys::neurons predicted({ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });
		CHECK(neurosys::loss::FnCost[neurosys::loss::squaredError](neurosys::loss::Fn[neurosys::loss::squaredError](observed, predicted))==Approx(0));
	}

}

/*
 __   __  _______  _______  ______    ___   __   __ 
|  |_|  ||   _   ||       ||    _ |  |   | |  |_|  |
|       ||  |_|  ||_     _||   | ||  |   | |       |
|       ||       |  |   |  |   |_||_ |   | |       |
|       ||       |  |   |  |    __  ||   |  |     | 
| ||_|| ||   _   |  |   |  |   |  | ||   | |   _   |
|_|   |_||__| |__|  |___|  |___|  |_||___| |__| |__|

*/

// mean, median, sum etc...
TEST_CASE("matrix sum", "[matrix][sum]")
{
	CHECK(neurosys::maths::sum(neurosys::matrix(0, 0))==0);
	CHECK(neurosys::maths::sum(neurosys::matrix(1, 1))==0);
	CHECK(neurosys::maths::sum(neurosys::matrix(2, 2))==0);
	CHECK(neurosys::maths::sum(neurosys::matrix(4, 4))==0);
	
	CHECK(neurosys::maths::sum(neurosys::matrix(std::vector<double>({1.0}), 1))==1.0);
	CHECK(neurosys::maths::sum(neurosys::matrix(std::vector<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }), 1))==21.0);
	CHECK(neurosys::maths::sum(neurosys::matrix(std::vector<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }), 2))==21.0);
	CHECK(neurosys::maths::sum(neurosys::matrix(std::vector<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }), 3))==21.0);
	
	CHECK(neurosys::maths::sum(neurosys::matrix(std::vector<double>({ 0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1 }), 1))==2.5);
	CHECK(neurosys::maths::sum(neurosys::matrix(std::vector<double>({ 0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0 }), 1))==2.5);

	//CHECK(neurosys::maths::sum(neurosys::matrix(std::vector<double>({ -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5 }), 1))==Approx(0.0));
	CHECK(ApproxEquals(neurosys::maths::sum(neurosys::matrix(std::vector<double>({ -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5 }), 1)),0.0));
}

TEST_CASE("matrix mean", "[matrix][mean]")
{
	CHECK(neurosys::maths::mean(neurosys::matrix(0, 0))==0);
	CHECK(neurosys::maths::mean(neurosys::matrix(1, 1))==0);
	CHECK(neurosys::maths::mean(neurosys::matrix(2, 2))==0);
	CHECK(neurosys::maths::mean(neurosys::matrix(4, 4))==0);
	
	CHECK(neurosys::maths::mean(neurosys::matrix(std::vector<double>({1.0}), 1))==1.0);
	CHECK(neurosys::maths::mean(neurosys::matrix(std::vector<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }), 1))==Approx(3.5));
	CHECK(neurosys::maths::mean(neurosys::matrix(std::vector<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }), 2))==Approx(3.5));
	CHECK(neurosys::maths::mean(neurosys::matrix(std::vector<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }), 3))==Approx(3.5));
	
	CHECK(neurosys::maths::mean(neurosys::matrix(std::vector<double>({ 0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1 }), 1))==Approx(0.277777));
	CHECK(neurosys::maths::mean(neurosys::matrix(std::vector<double>({ 0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0 }), 1))==Approx(0.25));

	CHECK(ApproxEquals(neurosys::maths::mean(neurosys::matrix(std::vector<double>({ -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5 }), 1)), 0));
}

TEST_CASE("matrix largest", "[matrix][largest]")
{
	CHECK(neurosys::maths::largest(neurosys::matrix(0, 0))==0);
	CHECK(neurosys::maths::largest(neurosys::matrix(1, 1))==0);
	CHECK(neurosys::maths::largest(neurosys::matrix(2, 2))==0);
	CHECK(neurosys::maths::largest(neurosys::matrix(4, 4))==0);
	
	CHECK(neurosys::maths::largest(neurosys::matrix(std::vector<double>({1.0}), 1))==0);
	CHECK(neurosys::maths::largest(neurosys::matrix(std::vector<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }), 1))==5);
	CHECK(neurosys::maths::largest(neurosys::matrix(std::vector<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }), 2))==5);
	CHECK(neurosys::maths::largest(neurosys::matrix(std::vector<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }), 3))==5);
	
	CHECK(neurosys::maths::largest(neurosys::matrix(std::vector<double>({ 0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1 }), 1))==4);
	CHECK(neurosys::maths::largest(neurosys::matrix(std::vector<double>({ 0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0 }), 1))==4);

	CHECK(neurosys::maths::largest(neurosys::matrix(std::vector<double>({ -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5 }), 1))==10);
}


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
		neurosys::matrix m({ { 1.0, 2.0 } }, 1);
		CHECK(m.size() == 2);
		CHECK(m.isColumnVector());
		CHECK(!m.isRowVector());
		CHECK(m.value(0, 0) == 1.0);
		CHECK(m.value(0, 1) == 2.0);
	}
	{
		neurosys::matrix m({ { 1.0, 2.0 } }, 1);
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
		neurosys::matrix result = neurosys::maths::multiply(m, n);

		CHECK(result.size() == 1);
		CHECK(result.value(0, 0) == 1.0);
	}
	{
		neurosys::matrix m({ {1.0, 2.0, 3.0 } }, 3);
		neurosys::matrix n({ {1.0, 2.0, 3.0 } }, 1);
		neurosys::matrix result = neurosys::maths::multiply(m, n);


		CHECK(result.size() == 1);
		CHECK(result.value(0, 0) == 14.0);
	}
	{
		neurosys::matrix m({ {1.0, 2.0, 3.0 } }, 1);
		neurosys::matrix n({ {1.0, 2.0, 3.0 } }, 3);
		neurosys::matrix result = neurosys::maths::multiply(m, n);

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
		neurosys::matrix result = neurosys::maths::multiply(m, n);

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

/*
 __    _  _______  _______  _     _  _______  ______    ___   _ 
|  |  | ||       ||       || | _ | ||       ||    _ |  |   | | |
|   |_| ||    ___||_     _|| || || ||   _   ||   | ||  |   |_| |
|       ||   |___   |   |  |       ||  | |  ||   |_||_ |      _|
|  _    ||    ___|  |   |  |       ||  |_|  ||    __  ||     |_ 
| | |   ||   |___   |   |  |   _   ||       ||   |  | ||    _  |
|_|  |__||_______|  |___|  |__| |__||_______||___|  |_||___| |_|

*/

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

TEST_CASE("network feedForward", "[network][observation]")
{
	// create a network, manually calculate the feedforward and then check...




}

TEST_CASE("network backPropagate", "[network][backPropagate]")
{
    
}


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
	neurosys::observation result = neurosys::feedForward(net, in);

    CHECK(result[1].value(0) == Approx(0.9866130822));
	CHECK(result[1].value(1) == Approx(0.9950331983));
    CHECK(result[2].value(0) == Approx(0.889550614));
    CHECK(result[2].value(1) == Approx(0.8004));
    
	// Perform the back prop.
	neurosys::output expected(neurosys::neurons(std::vector<double>({ 0.1, 0.05 })));
	neurosys::neurons error = neurosys::loss::FnPrime[neurosys::loss::squaredError](result.back(), expected.neurons());
	neurosys::network bp = neurosys::backPropagate(net, result, error, 0.01);
	
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

TEST_CASE("mattmazur", "[mattmazur]")
{
	// https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

	neurosys::network net(
		neurosys::input(2),
		{ neurosys::layer(2,neurosys::activation::sigmoid, 0.35) },
		neurosys::output(2, neurosys::activation::sigmoid, 0.6)
	);

	net[1].weight(0, 0) = 0.15;
	net[1].weight(0, 1) = 0.2;
	net[1].weight(1, 0) = 0.25;
	net[1].weight(1, 1) = 0.3;
	net[2].weight(0, 0) = 0.4;
	net[2].weight(0, 1) = 0.45;
	net[2].weight(1, 0) = 0.5;
	net[2].weight(1, 1) = 0.55;

	neurosys::input in({ 0.05, 0.1 });
	std::vector<neurosys::neurons> result = neurosys::feedForward(net, in);

    CHECK(result[1].value(0) == Approx(0.5944759307));
	CHECK(result[1].value(1) == Approx(0.5962826993));
	CHECK(result[2].value(0) == Approx(0.7569319153));
	CHECK(result[2].value(1) == Approx(0.7677178798));

	neurosys::output expected(neurosys::neurons(std::vector<double>({ 0.01, 0.99 })));
	
}

// https://medium.com/@rajatgupta310198/getting-started-with-neural-network-for-regression-and-tensorflow-58ad3bd75223
// regression example...

// https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html

// XOR ann  example...

TEST_CASE("xor", "[xor]")
{
	// https://towardsdatascience.com/implementing-the-xor-gate-using-backpropagation-in-neural-networks-c1f255b4f20d

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
	//for (unsigned int n = 0; n < 1000; ++n)
	//{
	//	net = neurosys::train(net, inputs, expected, neurosys::loss::squaredError, 0.1);
	//	std::cout << '\n';
	//}
	



}
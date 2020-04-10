
CXX=g++-9


build_test: ./test/test.cpp
	$(CXX) test/test.cpp -o bin/test

build_examples: ./examples/mnist/example1.cpp
	mkdir -p bin/mnist
	cp examples/mnist/*ubyte bin/mnist
	$(CXX) examples/mnist/example1.cpp -o bin/mnist/example1 

clean:
	rm -r bin

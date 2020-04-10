
CXX=g++-9


build_test: ./test/test.cpp
	$(CXX) test/test.cpp -o bin/test

build_example_xor: ./examples/xor/xor.cpp
	mkdir -p bin/xor
	$(CXX) examples/xor/xor.cpp -o bin/xor/xor 

build_examples: build_example_xor

clean:
	rm -r bin

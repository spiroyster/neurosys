#ifndef NEUROSYS_MNIST_HPP
#define NEUROSYS_MNIST_HPP

#include "neurosys.hpp"

namespace neurosys
{
	namespace MNIST
	{
		// load in the images into individual layers...
		// http://yann.lecun.com/exdb/mnist/

		typedef std::vector<neurosys::layer> Images;
		
		static std::vector<char> HandwritingLabelsRead(const std::string& filename)
		{
			std::ifstream file(filename.c_str(), std::ios::binary);
			if (!file)
				throw std::exception("Unable to read label file.");

			// Read header (mgic num)
			unsigned int value;
			file.read(reinterpret_cast<char *>(&value), sizeof(value));
			if (value != 2049)
				throw std::exception("Not MNIST label file.");

			// Read size of items...
			file >> value;

			// Read data...
			std::vector<char> result(
				(std::istreambuf_iterator<char>(file)),
				(std::istreambuf_iterator<char>()));

			return result;
		}

		static Images HandwritingImagesRead(const std::string& filename)
		{
			Images result;



			return result;
		}




	}
}


#endif // NEUROSYS_MNIST_HPP
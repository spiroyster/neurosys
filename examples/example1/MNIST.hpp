#ifndef NEUROSYS_MNIST_HPP
#define NEUROSYS_MNIST_HPP

#include "..\..\include\neurosys.hpp"

#include <array>
#include <fstream>
#include <filesystem>

namespace neurosys
{
	namespace MNIST
	{
		// load in the images into individual layers...
		// http://yann.lecun.com/exdb/mnist/

		typedef std::vector<neurosys::layer> Images;

		static int32_t readint32_t(std::ifstream& file, std::streampos pos)
		{
			char buffer[4];
			file.seekg(pos);
			file.read(reinterpret_cast<char *>(&buffer), sizeof(int32_t));

			return ((unsigned char)(buffer[0]) << 24 |
					(unsigned char)(buffer[1]) << 16 |
					(unsigned char)(buffer[2]) << 8 |
					(unsigned char)(buffer[3]));
		}
		
		static std::vector<char> HandwritingLabelsRead(const std::string& filename)
		{
			std::ifstream file(filename.c_str(), std::ios::binary | std::ios::ate);
			if (!file)
				throw std::exception("Unable to read labels file.");

			// Check magic...
			int32_t magic = readint32_t(file, 0);
			if (magic != 2049)
				throw std::exception("Not MNIST labels file.");

			// Read number of items...
			int32_t count = readint32_t(file, sizeof(int32_t));

			// Read data...
			std::vector<char> result(count);
			file.seekg(sizeof(int32_t) * 2);
			file.read(reinterpret_cast<char *>(&result.front()), count);

			return result;
		}

		static Images HandwritingImagesRead(const std::string& filename)
		{
			std::ifstream file(filename.c_str(), std::ios::binary | std::ios::ate);
			if (!file)
				throw std::exception("Unable to read images file.");

			// Check magic...
			int32_t magic = readint32_t(file, 0);
			if (magic != 2051)
				throw std::exception("Not MNIST images file.");

			// Read number of items...
			int32_t count = readint32_t(file, sizeof(int32_t));

			// Read row count and column count...
			int32_t rowCount = readint32_t(file, sizeof(int32_t) * 2);
			int32_t columnCount = readint32_t(file, sizeof(int32_t) * 3);
			int32_t imagePixelCount = rowCount * columnCount;

			// Read data...
			std::vector<unsigned char> pixels(count * imagePixelCount);
			file.seekg(sizeof(int32_t) * 4);
			file.read(reinterpret_cast<char *>(&pixels.front()), pixels.size());

			// convert to the neurosys layer type...
			Images result(count, neurosys::layer(imagePixelCount, neurosys::activation::fastSigmoid));
			for (int n = 0; n < count; ++n)
				for (int nn = 0; nn < imagePixelCount; ++nn)
					result[n].neurons_[nn] = static_cast<double>(pixels[n*imagePixelCount + nn]) / 255.0;
			
			return result;
		}




	}
}


#endif // NEUROSYS_MNIST_HPP
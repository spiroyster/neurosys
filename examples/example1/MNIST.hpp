#ifndef NEUROSYS_MNIST_HPP
#define NEUROSYS_MNIST_HPP

#include "../../include/neurosys.hpp"

#include <array>
#include <fstream>
#include <filesystem>
#include <stdexcept>

namespace neurosys
{
	namespace MNIST
	{
		// load in the images into individual layers...
		// http://yann.lecun.com/exdb/mnist/

		typedef std::vector<neurosys::input> Images;

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
		
		static std::vector<neurosys::output> HandwritingLabelsRead(const std::string& filename)
		{
			std::ifstream file(filename.c_str(), std::ios::binary | std::ios::ate);
			if (!file)
				throw std::runtime_error("Unable to read labels file.");

			// Check magic...
			int32_t magic = readint32_t(file, 0);
			if (magic != 2049)
				throw std::runtime_error("Not MNIST labels file.");

			// Read number of items...
			int32_t count = readint32_t(file, sizeof(int32_t));

			// Read data...
			std::vector<char> labels(count);
			file.seekg(sizeof(int32_t) * 2);
			file.read(reinterpret_cast<char *>(&labels.front()), count);

			std::vector<neurosys::output> result;

			return result;
		}

		static Images HandwritingImagesRead(const std::string& filename)
		{
			std::ifstream file(filename.c_str(), std::ios::binary | std::ios::ate);
			if (!file)
				throw std::runtime_error("Unable to read images file.");

			// Check magic...
			int32_t magic = readint32_t(file, 0);
			if (magic != 2051)
				throw std::runtime_error("Not MNIST images file.");

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
			Images result(count, neurosys::input(imagePixelCount));
			for (int n = 0; n < count; ++n)
				for (int nn = 0; nn < imagePixelCount; ++nn)
					result[n].neuron(nn) = static_cast<double>(pixels[n*imagePixelCount + nn]) / 255.0;
			
			return result;
		}




	}
}


#endif // NEUROSYS_MNIST_HPP

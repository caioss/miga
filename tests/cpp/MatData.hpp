#ifndef MATDATA_HPP
#define MATDATA_HPP

/*
 * Binary format
 *
 * All data is little-endian
 *
 * uint32_t * 1 : number of dimensions (N)
 * uint32_t * N : each dimension size
 * T[] : data
*/

#include <vector>
#include <cstdint>
#include <fstream>

template <class T>
class MatData
{
public:
    int32_t dims;
    std::vector<int32_t> shape;
    std::vector<T> data;

    MatData(const char* filename)
    {
        std::ifstream file(filename, std::ios::binary);

        // Stop eating new lines in binary mode!!!
        file.unsetf(std::ios::skipws);

        // Dimensions
        file.read((char *)&dims, sizeof(int32_t));
        shape.reserve(dims);
        file.read((char *)shape.data(), dims * sizeof(int32_t));

        // Data size
        size_t size { 1 };
        for (int i = 0; i != dims; ++i)
        {
            size *= shape[i];
        }
        data.resize(size);

        // Read the data
        file.read((char *)data.data(), size * sizeof(T));
        file.close();
    }

    T operator[](const size_t index) const
    {
        return data[index];
    }

    size_t size() const
    {
        return data.size();
    }

    const T *ptr() const
    {
        return data.data();
    }
};

#endif

#ifndef TEST_TOOLS_HPP
#define TEST_TOOLS_HPP

#include <cstddef>

template <class T, class U>
inline double vectors_equal(const T vec1, const U vec2, const size_t size, const size_t offset = 0)
{
    for (size_t i = 0; i != size; ++i)
    {
        if (vec1[offset * size + i] != vec2[offset * size + i])
        {
            return false;
        }
    }
    return true;
}

#endif

#pragma once

#include "cuda_includes.hpp"
#include <glm/glm.hpp>

inline int calculateNumBlocksPerGrid(int n, int blockSize)
{
    return (n + blockSize - 1) / blockSize;
}

inline dim3 calculateNumBlocksPerGrid(const int res, const dim3& blockSize)
{
    return dim3(calculateNumBlocksPerGrid(res, blockSize.x));
}

inline dim3 calculateNumBlocksPerGrid(const glm::ivec2& res, const dim3& blockSize)
{
    return dim3(calculateNumBlocksPerGrid(res.x, blockSize.x), calculateNumBlocksPerGrid(res.y, blockSize.y));
}

inline dim3 calculateNumBlocksPerGrid(const glm::ivec3& res, const dim3& blockSize)
{
    return dim3(calculateNumBlocksPerGrid(res.x, blockSize.x), calculateNumBlocksPerGrid(res.y, blockSize.y), calculateNumBlocksPerGrid(res.z, blockSize.z));
}

#pragma once

#include "cuda_includes.hpp"
#include <glm/glm.hpp>

inline dim3 calculateBlocksPerGrid(const glm::ivec2& res, const dim3& blockSize)
{
    return dim3((res.x + blockSize.x - 1) / blockSize.x, (res.y + blockSize.y - 1) / blockSize.y);
}

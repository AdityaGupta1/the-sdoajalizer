#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

struct ResolutionHash
{
    size_t operator()(const glm::ivec2& k)const
    {
        return std::hash<int>()(k.x) ^ std::hash<int>()(k.y);
    }
};

struct Texture
{
    cudaArray_t pixelArray;
    cudaTextureObject_t textureObj;
    glm::ivec2 resolution;
    int numReferences{ 0 };
};
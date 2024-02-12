#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

#include <functional>

struct ResolutionHash
{
    size_t operator()(const glm::ivec2& k) const
    {
        return std::hash<int>()(k.x) ^ std::hash<int>()(k.y);
    }
};

struct Texture
{
    glm::vec4* dev_pixels{ nullptr };
    glm::ivec2 resolution{ 0, 0 };
    int numReferences{ 0 };
    glm::vec4 singleColor{ 0, 0, 0, 1 };

    static Texture nullCheck(Texture* inTex);

    void setColor(glm::vec4 col);
    void setColor(glm::vec3 col);
    void setColor(float col);

    __host__ __device__ inline bool isSingleColor()
    {
        return this->resolution.x == 0;
    }
};

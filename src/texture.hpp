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

    void setSingleColor(glm::vec4 col);
    void setSingleColor(glm::vec3 col);
    void setSingleColor(float col);

    __host__ __device__ inline bool isSingleColor()
    {
        return this->resolution.x == 0;
    }

    __host__ __device__ inline glm::vec4 getColor(int x, int y, glm::vec4 backup = glm::vec4(0, 0, 0, 1))
    {
        if (isSingleColor())
        {
            return singleColor;
        }
        else
        {
            if (x < resolution.x && y < resolution.y)
            {
                int idx = y * resolution.x + x;
                return dev_pixels[idx];
            }
            else
            {
                return backup;
            }
        }
    }

    static glm::ivec2 getFirstResolution(std::initializer_list<Texture*> textures);
};

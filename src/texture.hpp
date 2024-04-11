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

    __device__ inline glm::vec4 getColorClamp(int x, int y, glm::vec4 backup = glm::vec4(0, 0, 0, 1))
    {
        if (isSingleColor())
        {
            return singleColor;
        }

        if (x < resolution.x && y < resolution.y)
        {
            return dev_pixels[y * resolution.x + x];
        }

        return backup;
    }

    __device__ inline glm::vec4 getColorReplicate(int x, int y)
    {
        if (isSingleColor())
        {
            return singleColor;
        }

        x = glm::clamp(x, 0, resolution.x - 1);
        y = glm::clamp(y, 0, resolution.y - 1);
        return dev_pixels[y * resolution.x + x];
    }

    __device__ inline void setColor(int x, int y, glm::vec4 col)
    {
        dev_pixels[y * resolution.x + x] = col;
    }

    static glm::ivec2 getFirstResolution(std::initializer_list<Texture*> textures);
};

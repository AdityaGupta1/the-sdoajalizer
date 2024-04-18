#pragma once

#include <glm/glm.hpp>
#include "cuda_includes.hpp"

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
private:
    glm::vec4* dev_pixels{ nullptr };

public:
    glm::ivec2 resolution{ 0, 0 };
    int numReferences{ 0 };
    glm::vec4 singleColor{ 0, 0, 0, 1 };

    void malloc(glm::ivec2 resolution);
    void free();

    glm::vec4* getDevPixels() const;
    __host__ __device__ inline bool hasDevPixels() const
    {
        return dev_pixels != nullptr;
    }

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

    __device__ inline glm::vec4 getColor(int idx)
    {
        return dev_pixels[idx];
    }

    __device__ inline glm::vec4 getColor(int x, int y)
    {
        return getColor(y * resolution.x + x);
    }

    __device__ inline void setColor(int idx, glm::vec4 col)
    {
        dev_pixels[idx] = col;
    }

    __device__ inline void setColor(int x, int y, glm::vec4 col)
    {
        setColor(y * resolution.x + x, col);
    }

    static glm::ivec2 getFirstResolution(std::initializer_list<Texture*> textures);
};

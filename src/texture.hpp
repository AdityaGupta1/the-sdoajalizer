#pragma once

#include <glm/glm.hpp>
#include "cuda_includes.hpp"

#include "color_utils.hpp"

#include <functional>

struct ResolutionHash
{
    size_t operator()(const glm::ivec2& k) const
    {
        return std::hash<int>()(k.x) ^ std::hash<int>()(k.y);
    }
};

enum class TextureType
{
    SINGLE, MULTI
};

struct Texture
{
public:
    __host__ __device__ static inline glm::vec4 singleToMulti(float single)
    {
        return glm::vec4(single, single, single, 1.f);
    }

    __host__ __device__ static inline float multiToSingle(glm::vec4 multi)
    {
        return ColorUtils::luminance(glm::vec3(multi));
    }

private:
    float* dev_pixelsSingle{ nullptr };
    glm::vec4* dev_pixelsMulti{ nullptr };
    glm::vec4 uniformColor{ 0, 0, 0, 1 };

public:
    glm::ivec2 resolution{ 0, 0 };
    int numReferences{ 0 };

    template<TextureType type>
    __host__ inline void malloc(glm::ivec2 resolution)
    {
        this->resolution = resolution;
        if constexpr (type == TextureType::SINGLE)
        {
            CUDA_CHECK(cudaMalloc(&dev_pixelsSingle, resolution.x * resolution.y * sizeof(float)));
        }
        else
        {
            CUDA_CHECK(cudaMalloc(&dev_pixelsMulti, resolution.x * resolution.y * sizeof(glm::vec4)));
        }
    }

    __host__ inline void free()
    {
        CUDA_CHECK(cudaFree(dev_pixelsSingle));
        CUDA_CHECK(cudaFree(dev_pixelsMulti));
    }

    template<TextureType type>
    __host__ __device__ auto getDevPixels() const
    {
        if constexpr (type == TextureType::SINGLE)
        {
            return dev_pixelsSingle;
        }
        else
        {
            return dev_pixelsMulti;
        }
    }

    template<TextureType type>
    __host__ __device__ inline bool isType() const
    {
        if constexpr (type == TextureType::SINGLE)
        {
            return dev_pixelsSingle != nullptr;
        }
        else
        {
            return dev_pixelsMulti != nullptr;
        }
    }

    template<TextureType type>
    __host__ __device__ auto getUniformColor()
    {
        return convertTo<type>(uniformColor);
    }
    void setUniformColor(glm::vec4 col);
    void setUniformColor(glm::vec3 col);
    void setUniformColor(float col);

    __host__ __device__ inline bool isUniform()
    {
        return this->resolution.x == 0;
    }

private:
    template<TextureType type>
    __host__ __device__ inline auto convertTo(float inCol)
    {
        if constexpr (type == TextureType::SINGLE)
        {
            return inCol;
        }
        else
        {
            return singleToMulti(inCol);
        }
    }

    template<TextureType type>
    __host__ __device__ inline auto convertTo(glm::vec4 inCol)
    {
        if constexpr (type == TextureType::MULTI)
        {
            return inCol;
        }
        else
        {
            return multiToSingle(inCol);
        }
    }

public:
    template<TextureType type>
    __device__ inline auto getColor(int idx)
    {
        if (dev_pixelsSingle != nullptr)
        {
            return convertTo<type>(dev_pixelsSingle[idx]);
        }
        else
        {
            return convertTo<type>(dev_pixelsMulti[idx]);
        }
    }

    template<TextureType type>
    __device__ inline auto getColor(int x, int y)
    {
        return getColor<type>(y * resolution.x + x);
    }

    template<TextureType type>
    __device__ inline void setColor(int idx, auto col)
    {
        getDevPixels<type>()[idx] = convertTo<type>(col);
    }

    template<TextureType type>
    __device__ inline void setColor(int x, int y, auto col)
    {
        setColor<type>(y * resolution.x + x, col);
    }

    template<TextureType type>
    __device__ inline auto getColorClamp(int x, int y, glm::vec4 backup = glm::vec4(0, 0, 0, 1))
    {
        if (isUniform())
        {
            return convertTo<type>(uniformColor);
        }

        if (x < resolution.x && y < resolution.y)
        {
            return getColor<type>(x, y);
        }

        return convertTo<type>(backup);
    }

    template<TextureType type>
    __device__ inline auto getColorReplicate(int x, int y)
    {
        if (isUniform())
        {
            return convertTo<type>(uniformColor);
        }

        x = glm::clamp(x, 0, resolution.x - 1);
        y = glm::clamp(y, 0, resolution.y - 1);
        return getColor<type>(x, y);
    }

    static glm::ivec2 getFirstResolutionFromList(std::initializer_list<Texture*> textures);
};

#include "texture.hpp"

void Texture::malloc(glm::ivec2 resolution)
{
    this->resolution = resolution;
    CUDA_CHECK(cudaMalloc(&dev_pixels, resolution.x * resolution.y * sizeof(glm::vec4)));
}

void Texture::free()
{
    CUDA_CHECK(cudaFree(dev_pixels));
}

glm::vec4* Texture::getDevPixels() const
{
    return dev_pixels;
}

void Texture::setSingleColor(glm::vec4 col)
{
    this->singleColor = col;
}

void Texture::setSingleColor(glm::vec3 col)
{
    this->singleColor = glm::vec4(col, 1);
}

void Texture::setSingleColor(float col)
{
    this->singleColor = glm::vec4(col, col, col, 1);
}

glm::ivec2 Texture::getFirstResolution(std::initializer_list<Texture*> textures)
{
    for (const auto& tex : textures)
    {
        if (!tex->isSingleColor())
        {
            return tex->resolution;
        }
    }

    return glm::ivec2(0, 0);
}

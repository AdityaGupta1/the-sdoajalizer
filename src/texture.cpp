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

void Texture::setUniformColor(glm::vec4 col)
{
    this->uniformColor = col;
}

void Texture::setUniformColor(glm::vec3 col)
{
    this->uniformColor = glm::vec4(col, 1);
}

void Texture::setUniformColor(float col)
{
    this->uniformColor = glm::vec4(col, col, col, 1);
}

glm::ivec2 Texture::getFirstResolution(std::initializer_list<Texture*> textures)
{
    for (const auto& tex : textures)
    {
        if (!tex->isUniform())
        {
            return tex->resolution;
        }
    }

    return glm::ivec2(0, 0);
}

#include "texture.hpp"

Texture Texture::nullCheck(Texture* inTex)
{
    if (inTex == nullptr)
    {
        return { .dev_pixels = nullptr };
    }
    else
    {
        return *inTex;
    }
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

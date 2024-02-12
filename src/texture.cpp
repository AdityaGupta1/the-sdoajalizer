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

void Texture::setColor(glm::vec4 col)
{
    this->singleColor = col;
}

void Texture::setColor(glm::vec3 col)
{
    this->singleColor = glm::vec4(col, 1);
}

void Texture::setColor(float col)
{
    this->singleColor = glm::vec4(col, col, col, 1);
}

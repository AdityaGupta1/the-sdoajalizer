#include "texture.hpp"

void Texture::setUniformColor(glm::vec4 col)
{
    this->uniformColor = col;
}

void Texture::setUniformColor(float col)
{
    this->uniformColor = Texture::singleToMulti(col);
}

glm::ivec2 Texture::getFirstResolutionFromList(std::initializer_list<Texture*> textures)
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

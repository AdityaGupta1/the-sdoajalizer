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
#pragma once

#include "node.hpp"
#include "edge.hpp"
#include "texture.hpp"

#include <unordered_map>
#include <unordered_set>
#include <memory>

#include "cuda_includes.hpp"
#include <glm/glm.hpp>
#include <GL/glew.h>

class Node;
class Pin;
class Edge;

class NodeEvaluator
{
private:
    Node* outputNode{ nullptr };

    std::unordered_map<glm::ivec2, std::vector<std::unique_ptr<Texture>>, ResolutionHash> textures;
    Texture* outputTexture{ nullptr };

    std::vector<Texture*> requestedTextures;

public:
    GLuint viewerTex;

    const glm::ivec2 outputResolution;

    NodeEvaluator(glm::ivec2 outputResolution);
    ~NodeEvaluator();

    void init();

    void setOutputNode(Node* outputNode);

    template<TextureType texType>
    Texture* requestTexture(glm::ivec2 resolution)
    {
        if (this->textures.contains(resolution))
        {
            for (const auto& texture : this->textures[resolution])
            {
                if (texture->hasDevPixels<texType>() && texture->numReferences == 0)
                {
                    ++texture->numReferences;
                    requestedTextures.push_back(texture.get());
                    return texture.get();
                }
            }
        }
        else
        {
            this->textures[resolution] = std::vector<std::unique_ptr<Texture>>();
        }

        auto tex = std::make_unique<Texture>();

        if (resolution.x != 0)
        {
            tex->malloc<texType>(resolution);
        }

        Texture* texPtr = tex.get();
        this->textures[resolution].push_back(std::move(tex));

        ++texPtr->numReferences;
        requestedTextures.push_back(texPtr);

        return texPtr;
    }

    template<TextureType texType>
    Texture* requestTexture() // defaults to output resolution
    {
        return this->requestTexture<texType>(this->outputResolution);
    }

    Texture* requestUniformTexture(); // resolution = (0, 0)

    Texture* getOutputTexture() const;
    void setOutputTexture(Texture* texture);
    bool hasOutputTexture() const;

    void setChangedNode(Node* changedNode);

    void evaluate();
};

#pragma once

#include "node.hpp"
#include "node_enums.hpp"
#include "texture.hpp"

#include <unordered_set>
#include <string>

class Edge;
class Node;

class Pin
{
private:
    Node* node{ nullptr };
    std::unordered_set<Edge*> edges;

    bool canConnect{ true };
    TextureType textureType{ TextureType::MULTI };

    PinCacheState cacheState{ PinCacheState::NO_CACHE };
    Texture* cachedTexture{ nullptr };

public:
    const int id;
    const PinType pinType;
    const std::string name;

    Pin(int id, Node* node, PinType pinType, const std::string& name);

    Node* getNode() const;
    const std::unordered_set<Edge*>& getEdges() const;
    bool hasEdge() const;

    void addEdge(Edge* edge);
    void removeEdge(Edge* edge);
    void clearEdges();

    // utility function to get single texture for input pins (nullptr if no connected edge)
    Texture* getSingleTexture() const;

    void propagateTexture(Texture* texture);
    void clearTextures();

    Pin& setNoConnect();
    bool getCanConnect() const;

    Pin& setSingleChannel();
    TextureType getTextureType() const;

    unsigned int getColor() const;
    unsigned int getHoveredColor() const;

    PinCacheState getCacheState() const;
    Texture* getCachedTexture() const;
    void prepareForCache();
    void deleteCache();
};

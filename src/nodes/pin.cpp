#include "pin.hpp"

Pin::Pin(int id, Node* node, PinType pinType, const std::string& name)
    : id(id), node(node), pinType(pinType), name(name)
{}

Node* Pin::getNode() const
{
    return this->node;
}

const std::unordered_set<Edge*>& Pin::getEdges() const
{
    return this->edges;
}

bool Pin::hasEdge() const
{
    return !getEdges().empty();
}

void Pin::addEdge(Edge* edge)
{
    this->edges.insert(edge);
}

void Pin::removeEdge(Edge* edge)
{
    this->edges.erase(edge);
}

void Pin::clearEdges()
{
    this->edges.clear();
}

Texture* Pin::getSingleTexture() const
{
    if (this->edges.empty())
    {
        return nullptr;
    }

    return (*edges.begin())->getTexture();
}

void Pin::propagateTexture(Texture* texture)
{
    for (auto& edge : this->edges)
    {
        if (!edge->endPin->getNode()->getIsBeingEvaluated())
        {
            continue;
        }

        edge->setTexture(texture);
    }

    if (this->cachedTexture != nullptr)
    {
        printf("WARNING: calling propagateTexture() when cachedTexture != nullptr\n");
        --this->cachedTexture->numReferences;
        this->cachedTexture = nullptr;
    }

    if (this->cacheState == PinCacheState::PREPARED)
    {
        this->cachedTexture = texture;
        ++this->cachedTexture->numReferences;
        this->cacheState = PinCacheState::CACHED;
    }
}

void Pin::clearTextures()
{
    for (auto& edge : this->edges)
    {
        edge->clearTexture();
    }
}

Pin& Pin::setNoConnect()
{
    this->canConnect = false;
    return *this;
}

bool Pin::getCanConnect() const
{
    return this->canConnect;
}

Pin& Pin::setSingleChannel()
{
    this->textureType = TextureType::SINGLE;
    return *this;
}

TextureType Pin::getTextureType() const
{
    return this->textureType;
}

unsigned int Pin::getColor() const
{
    return textureType == TextureType::SINGLE ? IM_COL32(175, 175, 175, 180) : IM_COL32(53, 150, 250, 180);
}

unsigned int Pin::getHoveredColor() const
{
    return textureType == TextureType::SINGLE ? IM_COL32(175, 175, 175, 255) : IM_COL32(53, 150, 250, 255);
}

PinCacheState Pin::getCacheState() const
{
    return this->cacheState;
}

Texture* Pin::getCachedTexture() const
{
    return this->cachedTexture;
}

void Pin::prepareForCache()
{
    if (this->cacheState != PinCacheState::CACHED)
    {
        this->cacheState = PinCacheState::PREPARED;
    }
}

void Pin::deleteCache()
{
    this->cacheState = PinCacheState::NO_CACHE;

    if (this->cachedTexture != nullptr)
    {
        --this->cachedTexture->numReferences;
        this->cachedTexture = nullptr;
    }
}
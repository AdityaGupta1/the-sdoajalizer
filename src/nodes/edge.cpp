#include "edge.hpp"

int Edge::nextId = 1 << 24;

Edge::Edge(Pin* startPin, Pin* endPin)
    : id(Edge::nextId++), startPin(startPin), endPin(endPin)
{}

Texture* Edge::getTexture() const
{
    Texture* startPinCachedTexture = startPin->getCachedTexture();
    if (startPinCachedTexture != nullptr)
    {
        return startPinCachedTexture;
    }

    return this->texture;
}

void Edge::setTexture(Texture* texture)
{
    this->texture = texture;
    if (this->texture != nullptr)
    {
        ++this->texture->numReferences;
    }
}

void Edge::clearTexture()
{
    if (this->texture != nullptr)
    {
        --this->texture->numReferences;
        this->texture = nullptr;
    }
}

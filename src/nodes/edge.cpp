#include "edge.hpp"

int Edge::nextId = 1 << 24;

Edge::Edge(Pin* startPin, Pin* endPin)
    : id(Edge::nextId++), startPin(startPin), endPin(endPin)
{}

Texture* Edge::getTexture() const
{
    return this->texture;
}

void Edge::setTexture(Texture* texture)
{
    this->texture = texture;
    ++this->texture->numReferences;
}

void Edge::clearTexture()
{
    --this->texture->numReferences;
    this->texture = nullptr;
}
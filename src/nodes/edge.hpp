#pragma once

#include "node.hpp"
#include "node_evaluator.hpp"
#include "../texture.hpp"

class Node;
class Pin;
class NodeEvaluator;

class Edge
{
private:
    static int nextId;

    Texture* texture;

public:
    const int id;

    Pin* const startPin;
    Pin* const endPin;

    Edge(Pin* startPin, Pin* endPin);

    Texture* getTexture() const;
    void setTexture(Texture* texture);
    void clearTexture();
};
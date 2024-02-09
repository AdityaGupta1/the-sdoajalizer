#pragma once

#include "node.hpp"

class Node;
class Pin;

class Edge
{
private:
    static int nextId;

public:
    const int id;

    Pin* const startPin;
    Pin* const endPin;

    Edge(Pin* startPin, Pin* endPin);
};
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

    const Pin* startPin;
    const Pin* endPin;

    Edge(Pin* startPin, Pin* endPin);
};
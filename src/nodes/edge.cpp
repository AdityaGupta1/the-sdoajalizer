#include "edge.hpp"

int Edge::nextId = 1 << 24;

Edge::Edge(Pin* startPin, Pin* endPin)
    : id(Edge::nextId++), startPin(startPin), endPin(endPin)
{}
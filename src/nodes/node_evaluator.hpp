#pragma once

#include "node.hpp"
#include "edge.hpp"

class Node;
class Pin;
class Edge;

class NodeEvaluator
{
private:
    Node* outputNode{ nullptr };

public:
    NodeEvaluator();

    void setOutputNode(Node* outputNode);

    void evaluate();
};
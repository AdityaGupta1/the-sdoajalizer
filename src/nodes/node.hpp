#pragma once

#define NODE_ID_STRIDE 32

#include "node.hpp"

#include <vector>
#include <string>

class Edge;
class Node;

class Pin
{
private:
    Node* node;
    Edge* edge{ nullptr };

public:
    const int id;

    Pin(int id, Node* node);

    Node* getNode();
    Edge* getEdge();

    void setEdge(Edge* edge);
    void clearEdge();
};

class Node
{
private:
    static int nextId;

protected:
    const std::string name;

    Node(std::string name);

    void addPins(int numInput, int numOutput);

public:
    const int id;

    std::vector<Pin> inputPins;
    std::vector<Pin> outputPins;

    Pin& getPin(int pinId);

    void draw() const;
};
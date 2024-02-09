#pragma once

#define NODE_ID_STRIDE 32

#include "node.hpp"

#include <vector>
#include <string>
#include <unordered_set>

class Edge;
class Node;

class Pin
{
private:
    Node* node;
    std::unordered_set<Edge*> edges;

public:
    const int id;

    Pin(int id, Node* node);

    Node* getNode() const;
    const std::unordered_set<Edge*>& getEdges() const;

    void addEdge(Edge* edge);
    void removeEdge(Edge* edge);
    void clearEdges();
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
#pragma once

#define NODE_ID_STRIDE 32

#include "node.hpp"
#include "node_evaluator.hpp"
#include "../texture.hpp"

#include "../ImGui/imgui.h"

#include <vector>
#include <string>
#include <unordered_set>

class Edge;
class Node;
class NodeEvaluator;

class Pin
{
private:
    Node* node{ nullptr };
    std::unordered_set<Edge*> edges;

public:
    const int id;

    Pin(int id, Node* node);

    Node* getNode() const;
    const std::unordered_set<Edge*>& getEdges() const;

    void addEdge(Edge* edge);
    void removeEdge(Edge* edge);
    void clearEdges();

    // utility function to get single texture for input pins (nullptr if no connected edge)
    Texture* getSingleTexture() const;

    void propagateTexture(Texture* texture);
    void clearTextures();
};

class Node
{
    friend class NodeEvaluator;

private:
    static int nextId;

    bool isSelected{ false };

    void setIsSelected(bool isSelected);

protected:
    const std::string name;

    NodeEvaluator* nodeEvaluator{ nullptr };

    Node(std::string name);

    void addPins(int numInput, int numOutput);

    virtual unsigned int getTitleBarColor() const;
    virtual unsigned int getTitleBarSelectedColor() const;

    virtual void evaluate() = 0;
    void clearInputTextures();

    bool getIsSelected() const;

public:
    const int id;

    std::vector<Pin> inputPins;
    std::vector<Pin> outputPins;

    Pin& getPin(int pinId);

    void setNodeEvaluator(NodeEvaluator* nodeEvaluator);

    void draw() const;
};
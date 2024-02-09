#include "node.hpp"

#include "../ImGui/imnodes.h"


Pin::Pin(int id, Node* node)
    : id(id), node(node)
{}

Node* Pin::getNode() const
{
    return this->node;
}

const std::unordered_set<Edge*>& Pin::getEdges() const
{
    return this->edges;
}

void Pin::addEdge(Edge* edge)
{
    this->edges.insert(edge);
}

void Pin::removeEdge(Edge* edge)
{
    this->edges.erase(edge);
}

void Pin::clearEdges()
{
    this->edges.clear();
}

int Node::nextId = 0;

Node::Node(std::string name)
    : name(name), id(Node::nextId)
{
    Node::nextId += NODE_ID_STRIDE;
}

void Node::addPins(int numInput, int numOutput)
{
    int nextPinId = this->id + 1;

    for (int i = 0; i < numInput; ++i)
    {
        this->inputPins.push_back(Pin(nextPinId++, this));
    }

    for (int i = 0; i < numOutput; ++i)
    {
        this->outputPins.push_back(Pin(nextPinId++, this));
    }
}

Pin& Node::getPin(int pinId)
{
    int localPinId = pinId - this->id - 1;
    int numInputPins = inputPins.size();
    if (localPinId < numInputPins)
    {
        return inputPins[localPinId];
    }
    else
    {
        return outputPins[localPinId - numInputPins];
    }
}

void Node::draw() const
{
    ImNodes::BeginNode(this->id);

    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted(this->name.c_str());
    ImNodes::EndNodeTitleBar();

    for (const auto& inputPin : inputPins)
    {
        ImNodes::BeginInputAttribute(inputPin.id);
        ImGui::Text("input");
        ImNodes::EndInputAttribute();
    }

    for (const auto& outputPin : outputPins)
    {
        ImNodes::BeginOutputAttribute(outputPin.id);
        ImGui::Text("output");
        ImNodes::EndOutputAttribute();
    }

    ImNodes::EndNode();
}
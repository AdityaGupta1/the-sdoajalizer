#pragma once

#include "nodes/node.hpp"

class NodeMix : public Node
{
private:
    glm::vec4 backupCol1{ 0.5f, 0.5f, 0.5f, 1.0f };
    glm::vec4 backupCol2{ 0.5f, 0.5f, 0.5f, 1.0f };
    float factor{ 0.5f };

public:
    NodeMix();

protected:
    bool drawInputPinExtras(const Pin* pin, int pinNumber) override;
    void evaluate() override;
};
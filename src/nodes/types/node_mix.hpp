#pragma once

#include "nodes/node.hpp"

class NodeMix : public Node
{
private:
    glm::vec4 backupCol1{ Node::defaultBackupVec4 };
    glm::vec4 backupCol2{ Node::defaultBackupVec4 };
    float factor{ Node::defaultBackupFloat };

public:
    NodeMix();

protected:
    bool drawInputPinExtras(const Pin* pin, int pinNumber) override;
    void evaluate() override;
};
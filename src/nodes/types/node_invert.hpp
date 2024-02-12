#pragma once

#include "nodes/node.hpp"

class NodeInvert : public Node
{
private:
    glm::vec4 backupCol{ Node::defaultBackupVec4 };

public:
    NodeInvert();

protected:
    void evaluate() override;
};
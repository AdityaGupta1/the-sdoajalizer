#pragma once

#include "nodes/node.hpp"

class NodeInvert : public Node
{
private:
    glm::vec4 backupCol{ NodeUI::defaultBackupVec4 };

public:
    NodeInvert();

protected:
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void evaluate() override;
};

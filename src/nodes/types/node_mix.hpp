#pragma once

#include "nodes/node.hpp"

class NodeMix : public Node
{
private:
    glm::vec4 backupCol1{ NodeUI::defaultBackupVec4 };
    glm::vec4 backupCol2{ NodeUI::defaultBackupVec4 };
    float backupFactor{ NodeUI::defaultBackupFloat };

public:
    NodeMix();

protected:
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void evaluate() override;
};

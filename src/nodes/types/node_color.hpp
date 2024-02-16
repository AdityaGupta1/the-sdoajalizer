#pragma once

#include "nodes/node.hpp"

class NodeColor : public Node
{
private:
    glm::vec4 backupCol{ NodeUI::defaultBackupVec4 };

public:
    NodeColor();

protected:
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void evaluate() override;

    std::string debugGetSrcFileName() const override;
};

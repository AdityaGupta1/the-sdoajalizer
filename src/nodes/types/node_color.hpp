#pragma once

#include "nodes/node.hpp"

class NodeColor : public Node
{
private:
    struct
    {
        glm::vec4 color{ NodeUI::defaultBackupVec4 };
    } constParams;

public:
    NodeColor();

protected:
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void evaluate() override;
};

#pragma once

#include "nodes/node.hpp"

class NodeBrightnessContrast : public Node
{
private:
    struct
    {
        glm::vec4 color{ NodeUI::defaultBackupVec4 };
        float brightness{ 0.f };
        float contrast{ 0.f };
    } constParams;

public:
    NodeBrightnessContrast();

protected:
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void evaluate() override;
};

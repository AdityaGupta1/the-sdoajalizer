#pragma once

#include "nodes/node.hpp"

class NodeMix : public Node
{
private:
    struct
    {
        bool clamp{ false };
        float factor{ NodeUI::defaultBackupFloat };
        glm::vec4 color1{ NodeUI::defaultBackupVec4 };
        glm::vec4 color2{ NodeUI::defaultBackupVec4 };
    } constParams;

public:
    NodeMix();

protected:
    bool drawPinBeforeExtras(const Pin* pin, int pinNumber) override;
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void _evaluate() override;
};

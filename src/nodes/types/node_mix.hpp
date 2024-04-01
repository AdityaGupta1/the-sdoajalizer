#pragma once

#include "nodes/node.hpp"

class NodeMix : public Node
{
private:
    struct
    {
        glm::vec4 color1{ NodeUI::defaultBackupVec4 };
        glm::vec4 color2{ NodeUI::defaultBackupVec4 };
        float factor{ NodeUI::defaultBackupFloat };
    } constParams;

public:
    NodeMix();

protected:
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void evaluate() override;

    std::string debugGetSrcFileName() const override;
};

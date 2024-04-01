#pragma once

#include "nodes/node.hpp"

class NodeInvert : public Node
{
private:
    struct
    {
        glm::vec4 color{ NodeUI::defaultBackupVec4 };
    } constParams;

public:
    NodeInvert();

protected:
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void evaluate() override;

    std::string debugGetSrcFileName() const override;
};

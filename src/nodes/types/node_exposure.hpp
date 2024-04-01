#pragma once

#include "nodes/node.hpp"

class NodeExposure : public Node
{
private:
    struct
    {
        glm::vec4 color{ NodeUI::defaultBackupVec4 };
        float exposure{ 0.f };
    } constParams;

public:
    NodeExposure();

protected:
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void evaluate() override;

    std::string debugGetSrcFileName() const override;
};

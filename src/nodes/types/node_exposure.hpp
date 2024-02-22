#pragma once

#include "nodes/node.hpp"

class NodeExposure : public Node
{
private:
    float backupExposure{ 0.f };
    glm::vec4 backupCol{ NodeUI::defaultBackupVec4 };

public:
    NodeExposure();

protected:
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void evaluate() override;

    std::string debugGetSrcFileName() const override;
};

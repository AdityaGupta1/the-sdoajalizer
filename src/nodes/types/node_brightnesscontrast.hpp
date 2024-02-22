#pragma once

#include "nodes/node.hpp"

class NodeBrightnessContrast : public Node
{
private:
    glm::vec4 backupCol{ NodeUI::defaultBackupVec4 };
    float backupBrightness{ 0.f };
    float backupContrast{ 0.f };

public:
    NodeBrightnessContrast();

protected:
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void evaluate() override;

    std::string debugGetSrcFileName() const override;
};

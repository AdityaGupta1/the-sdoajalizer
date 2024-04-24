#pragma once

#include "nodes/node.hpp"

enum class ComponentsType
{
    RGB, HSV
};

template<ComponentsType componentsType>
class NodeSeparateComponents : public Node
{
private:
    struct
    {
        glm::vec4 color{ NodeUI::defaultBackupVec4 };
    } constParams;

public:
    NodeSeparateComponents(const std::string& name);

protected:
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void _evaluate() override;
};

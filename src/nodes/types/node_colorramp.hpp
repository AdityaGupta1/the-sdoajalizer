#pragma once

#include "nodes/node.hpp"

#include <imgui_gradient/imgui_gradient.hpp>

class NodeColorRamp : public Node
{
private:
    ImGG::GradientWidget gradientWidget{};

    struct
    {
        float factor{ 0.5f };
    } constParams;

public:
    NodeColorRamp();

protected:
    bool drawPinBeforeExtras(const Pin* pin, int pinNumber) override;
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void _evaluate() override;
};

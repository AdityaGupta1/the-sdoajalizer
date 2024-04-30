#pragma once

#include "nodes/node.hpp"

#include <imgui_gradient/imgui_gradient.hpp>

struct InterpolationName
{
    const ImGG::Interpolation interpolation;
    const std::string name;
};

class NodeColorRamp : public Node
{
private:
    static std::vector<InterpolationName> interpolationNames;

    ImGG::GradientWidget gradientWidget{};
    ImGG::RawMark* dev_rawMarks{ nullptr };

    struct
    {
        InterpolationName* interpolationNamePtr{ &interpolationNames[0] };
        float factor{ 0.5f };
    } constParams;

public:
    NodeColorRamp();
    ~NodeColorRamp() override;

protected:
    bool drawPinBeforeExtras(const Pin* pin, int pinNumber) override;
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void _evaluate() override;
};

#pragma once

#include "nodes/node.hpp"
#include "texture.hpp"

class NodeOutput : public Node
{
public:
    NodeOutput();

protected:
    unsigned int getTitleBarColor() const override;
    unsigned int getTitleBarHoveredColor() const override;

    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void _evaluate() override;
};

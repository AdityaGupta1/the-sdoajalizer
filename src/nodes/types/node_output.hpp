#pragma once

#include "../node.hpp"
#include "../../texture.hpp"

class NodeOutput : public Node
{
public:
    NodeOutput();

protected:
    unsigned int getTitleBarColor() const override;
    unsigned int getTitleBarSelectedColor() const override;

    void evaluate() override;
};

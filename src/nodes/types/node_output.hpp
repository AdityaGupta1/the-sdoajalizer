#pragma once

#include "../node.hpp"

class NodeOutput : public Node
{
public:
    NodeOutput();

protected:
    unsigned int getTitleBarColor() const override;
    unsigned int getTitleBarSelectedColor() const override;
};

#pragma once

#include "../node.hpp"

class NodeUvGradient : public Node
{
public:
    NodeUvGradient();

protected:
    void evaluate() override;
};
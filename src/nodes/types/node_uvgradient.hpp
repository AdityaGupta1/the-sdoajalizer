#pragma once

#include "nodes/node.hpp"

class NodeUvGradient : public Node
{
public:
    NodeUvGradient();

protected:
    void evaluate() override;
};
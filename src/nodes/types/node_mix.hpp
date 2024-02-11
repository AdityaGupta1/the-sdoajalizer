#pragma once

#include "nodes/node.hpp"

class NodeMix : public Node
{
public:
    NodeMix();

protected:
    void evaluate() override;
};
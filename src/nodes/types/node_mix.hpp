#pragma once

#include "../node.hpp"

class NodeMix : public Node
{
public:
    NodeMix();

protected:
    void evaluate() override;
};
#pragma once

#include "../node.hpp"

class NodeNoise : public Node
{
public:
    NodeNoise();

protected:
    void evaluate() override;
};
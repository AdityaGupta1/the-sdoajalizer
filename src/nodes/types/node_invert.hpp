#pragma once

#include "../node.hpp"

class NodeInvert : public Node
{
public:
    NodeInvert();

protected:
    void evaluate() override;
};
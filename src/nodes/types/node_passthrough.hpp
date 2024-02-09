#pragma once

#include "../node.hpp"

class NodePassthrough : public Node
{
public:
    NodePassthrough();

protected:
    void evaluate() override;
};
#pragma once

#include "nodes/node.hpp"

class NodeDebug : public Node
{
private:
    static int debugNum;

public:
    NodeDebug();

protected:
    void evaluate() override;
};

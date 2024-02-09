#include "node_debug.hpp"

int NodeDebug::debugNum = 0;

NodeDebug::NodeDebug()
    : Node("debug " + std::to_string(NodeDebug::debugNum++))
{
    addPins(4, 4);
}

void NodeDebug::evaluate()
{
    // do nothing
}
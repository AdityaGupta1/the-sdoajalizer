#include "node_debug.hpp"

int NodeDebug::debugNum = 0;

NodeDebug::NodeDebug()
    : Node("debug " + std::to_string(NodeDebug::debugNum++))
{
    for (int i = 0; i < 4; ++i)
    {
        addPin(PinType::INPUT);
        addPin(PinType::OUTPUT);
    }
}

void NodeDebug::evaluate()
{
    // do nothing
}

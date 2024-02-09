#include "node_passthrough.hpp"

NodePassthrough::NodePassthrough()
    : Node("passthrough")
{
    addPins(4, 4);
}

void NodePassthrough::evaluate()
{
    // do nothing
}
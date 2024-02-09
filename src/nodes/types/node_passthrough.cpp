#include "node_passthrough.hpp"

NodePassthrough::NodePassthrough()
    : Node("passthrough")
{
    addPins(1, 1);
}
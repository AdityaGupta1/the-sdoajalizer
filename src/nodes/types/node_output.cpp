#include "node_output.hpp"

NodeOutput::NodeOutput()
    : Node("output")
{
    addPins(1, 0);
}

unsigned int NodeOutput::getTitleBarColor() const
{
    return IM_COL32(255, 85, 0, 255);
}

unsigned int NodeOutput::getTitleBarSelectedColor() const
{
    return IM_COL32(255, 128, 0, 255);
}

void NodeOutput::evaluate()
{
    nodeEvaluator->setOutputTexture(inputPins[0].getSingleTexture());
}

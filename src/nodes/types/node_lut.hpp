#pragma once

#include "nodes/node.hpp"

class NodeLUT : public Node
{
private:
    std::string filePath;
    bool needsReloadFile{ false };

public:
    NodeLUT();

protected:
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void evaluate() override;
};

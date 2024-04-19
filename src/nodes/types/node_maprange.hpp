#pragma once

#include "nodes/node.hpp"

class NodeMapRange : public Node
{
private:
    struct
    {
        float value{ 0.f };
        float oldMin{ 0.f };
        float oldMax{ 1.f };
        float newMin{ 0.f };
        float newMax{ 1.f };
    } constParams;

public:
    NodeMapRange();

protected:
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void _evaluate() override;
};

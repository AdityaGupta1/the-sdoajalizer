#pragma once

#include "nodes/node.hpp"

enum class Operation
{
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
    POWER,
    MAX,
    MIN
};

struct OperationName
{
    const Operation operation;
    const std::string name;
};

class NodeMath : public Node
{
private:
    static std::vector<OperationName> operationNames;

    struct
    {
        OperationName* operationNamePtr{ &operationNames[0] };
        float inputA{ 0.5f };
        float inputB{ 0.5f };
    } constParams;

public:
    NodeMath();

protected:
    bool drawPinBeforeExtras(const Pin* pin, int pinNumber) override;
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void _evaluate() override;
};

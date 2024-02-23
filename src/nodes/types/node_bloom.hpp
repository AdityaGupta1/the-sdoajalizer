#pragma once

#include "nodes/node.hpp"

class NodeBloom : public Node
{
private:
    float backupThreshold{ 1.f };
    int backupIterations{ 1 };

public:
    NodeBloom();

protected:
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void evaluate() override;

    std::string debugGetSrcFileName() const override;
};

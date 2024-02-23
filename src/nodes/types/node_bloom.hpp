#pragma once

#include "nodes/node.hpp"

class NodeBloom : public Node
{
private:
    static constexpr int sizeMin = 4;
    static constexpr int sizeMax = 8;

    float backupThreshold{ 1.f };
    int backupSize{ 5 };
    float backupMix{ 0.f };

public:
    NodeBloom();

protected:
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void evaluate() override;

    std::string debugGetSrcFileName() const override;
};

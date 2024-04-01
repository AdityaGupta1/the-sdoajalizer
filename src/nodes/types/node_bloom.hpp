#pragma once

#include "nodes/node.hpp"

#include <array>

class NodeBloom : public Node
{
private:
    static constexpr int sizeMin = 4;
    static constexpr int sizeMax = 8;

    static constexpr int numBloomKernels = sizeMax - sizeMin + 1;
    static std::array<float*, numBloomKernels> dev_bloomKernels;

    struct
    {
        float threshold{ 1.f };
        int size{ 5 };
        float mix{ 0.f };
    } constParams;

public:
    NodeBloom();

    static void freeDeviceMemory();

protected:
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void evaluate() override;

    std::string debugGetSrcFileName() const override;
};

#pragma once

#include "nodes/node.hpp"

#include <array>

class NodePaintinator : public Node
{
private:
    int backupNumStrokes{ 16 };
    int backupMinStrokeSize{ 5 };
    int backupMaxStrokeSize{ 50 };
    float backupSizeBias{ 0.7f };

    static bool hasLoadedBrushes;
    static cudaArray_t pixelArray;
    static cudaTextureObject_t textureObject;

public:
    NodePaintinator();

    static void freeDeviceMemory();

protected:
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void evaluate() override;

    std::string debugGetSrcFileName() const override;
};

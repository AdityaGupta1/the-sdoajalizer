#pragma once

#include "nodes/node.hpp"

#include <array>

class NodePaintinator : public Node
{
private:
    int backupNumStrokes{ 16 };
    int backupMinStrokeSize{ 20 };
    int backupMaxStrokeSize{ 250 };
    float backupSizeBias{ 1.3f };

    static bool hasLoadedBrushes;
    static cudaArray_t brushPixelArray;
    static cudaTextureObject_t brushTextureObj;

public:
    NodePaintinator();

    static void freeDeviceMemory();

protected:
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void evaluate() override;

    void loadBrushes();

    std::string debugGetSrcFileName() const override;
};

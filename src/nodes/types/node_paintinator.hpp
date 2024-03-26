#pragma once

#include "nodes/node.hpp"

#include <array>

class NodePaintinator : public Node
{
private:
    int backupMinStrokeSize{ 5 };
    int backupMaxStrokeSize{ 400 };
    // TODO: parameter for brush alpha

    static bool hasLoadedBrushes;
    static cudaArray_t brushPixelArray;
    static cudaTextureObject_t brushTextureObj;

public:
    NodePaintinator();

    static void freeDeviceMemory();

protected:
    void loadBrushes();

    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void evaluate() override;

    std::string debugGetSrcFileName() const override;
};

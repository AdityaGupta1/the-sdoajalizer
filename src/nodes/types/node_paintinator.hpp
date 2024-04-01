#pragma once

#include "nodes/node.hpp"

#include <array>

struct PaintStroke
{
    glm::ivec2 pos;
    glm::mat3 transform;
    glm::vec3 color;
    glm::vec2 cornerUv;
};

class NodePaintinator : public Node
{
private:
    int backupMinStrokeSize{ 5 };
    int backupMaxStrokeSize{ 200 };
    float backupGridSizeFactor{ 0.25f };
    float backupBlurKernelSizeFactor{ 0.3f };
    float backupNewStrokeThreshold{ 0.15f };
    float backupBrushAlpha{ 1.f };

    static bool hasLoadedBrushes;
    static cudaArray_t brushPixelArray;
    static cudaTextureObject_t brushTextureObj;

    PaintStroke* dev_strokes{ nullptr };
    int numDevStrokes{ 0 };

public:
    NodePaintinator();
    ~NodePaintinator() override;

    static void freeDeviceMemory();

protected:
    void loadBrushes();

    bool drawPinBeforeExtras(const Pin* pin, int pinNumber) override;
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void evaluate() override;

    std::string debugGetSrcFileName() const override;
};

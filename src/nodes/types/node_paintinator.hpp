#pragma once

#include "nodes/node.hpp"

#include <array>

struct PaintStroke
{
    glm::ivec2 pos;
    glm::mat2 matRotate;
    float scale;
    glm::vec3 color;
    glm::vec2 uv;
};

struct PaintStrokeComparator
{
    __host__ __device__ bool operator()(const PaintStroke& stroke1, const PaintStroke& stroke2)
    {
        return stroke1.scale < stroke2.scale;
    }
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

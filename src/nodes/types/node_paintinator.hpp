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

struct BrushesTexture
{
    bool isLoaded{ false };
    const std::string filePath;
    const std::string displayName;

    cudaArray_t pixelArray{ nullptr };
    cudaTextureObject_t textureObj;

    glm::vec2 scale{ 1.f, 1.f };

    BrushesTexture(const std::string& filePath, const std::string& displayName);

    void load();
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

    static BrushesTexture brushesTex;

    PaintStroke* dev_strokes{ nullptr };
    int numDevStrokes{ 0 };

public:
    NodePaintinator();
    ~NodePaintinator() override;

    static void freeDeviceMemory();

protected:
    bool drawPinBeforeExtras(const Pin* pin, int pinNumber) override;
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void evaluate() override;

    std::string debugGetSrcFileName() const override;
};

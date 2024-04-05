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

struct BrushTexture
{
    bool isLoaded{ false };
    const std::string filePath;
    const std::string displayName;

    cudaArray_t pixelArray{ nullptr };
    cudaTextureObject_t textureObj;

    glm::vec2 scale{ 1.f, 1.f };

    BrushTexture(const std::string& filePath, const std::string& displayName);

    void load();
};

class NodePaintinator : public Node
{
private:
    struct
    {
        BrushTexture* brushTexturePtr{ &brushTextures[0] };
        float brushAlpha{ 1.f };
        int minStrokeSize{ 5 };
        int maxStrokeSize{ 200 };
        float gridSizeFactor{ 0.25f };
        float blurKernelSizeFactor{ 0.3f };
        float newStrokeThreshold{ 0.15f };
    } constParams;

    static std::vector<BrushTexture> brushTextures;

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
};

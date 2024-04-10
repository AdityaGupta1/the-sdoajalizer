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

struct BrushParams
{
    float brushAlpha;
    int minStrokeSize;
    int maxStrokeSize;
    float gridSizeFactor;
    float blurKernelSizeFactor;
    float newStrokeThreshold;
};

class NodePaintinator : public Node
{
private:
    struct
    {
        BrushTexture* brushTexturePtr{ &brushTextures[0] };
        std::unordered_map<BrushTexture*, BrushParams> brushParamsMap;

        BrushParams& getBrushParams()
        {
            return brushParamsMap[brushTexturePtr];
        }
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

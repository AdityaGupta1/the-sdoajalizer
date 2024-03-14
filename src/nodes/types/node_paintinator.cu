#include "node_paintinator.hpp"

#include "cuda_includes.hpp"

NodePaintinator::NodePaintinator()
    : Node("paint-inator")
{
    addPin(PinType::OUTPUT, "image");

    addPin(PinType::INPUT, "image");
    addPin(PinType::INPUT, "num strokes").setNoConnect();
    addPin(PinType::INPUT, "min stroke size").setNoConnect();
    addPin(PinType::INPUT, "max stroke size").setNoConnect();

    setExpensive();
}

bool NodePaintinator::drawPinExtras(const Pin* pin, int pinNumber)
{
    if (pin->pinType == PinType::OUTPUT || pin->hasEdge())
    {
        return false;
    }

    switch (pinNumber)
    {
    case 0: // image
        return false;
    case 1: // num strokes
        ImGui::SameLine();
        return NodeUI::IntEdit(backupNumStrokes, 0.02f, 8, 18);
    case 2: // min stroke size
        ImGui::SameLine();
        return NodeUI::IntEdit(backupMinStrokeSize, 0.15f, 1, 250);
    case 3: // max stroke size
        ImGui::SameLine();
        return NodeUI::IntEdit(backupMaxStrokeSize, 0.15f, 1, 500);
    default:
        throw std::runtime_error("invalid pin number");
    }
}

struct PaintStroke
{
    glm::vec2 pos;
    float scale;
    glm::vec4 color;
};

__global__ void kernPaint(Texture outTex, PaintStroke* strokes, int numStrokes)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= outTex.resolution.x || y >= outTex.resolution.y)
    {
        return;
    }

    glm::vec4 thisColor = glm::vec4(0, 0, 0, 1);
    glm::vec2 thisPos = glm::vec2(x, y);
    for (int strokeIdx = 0; strokeIdx < numStrokes; ++strokeIdx)
    {
        const PaintStroke& stroke = strokes[strokeIdx];

        if (glm::distance(thisPos, stroke.pos) <= stroke.scale)
        {
            thisColor = stroke.color;
            break;
        }
    }

    int idx = y * outTex.resolution.x + x;
    outTex.dev_pixels[idx] = thisColor;
}

void NodePaintinator::evaluate()
{
    Texture* inTex = getPinTextureOrSingleColor(inputPins[0], glm::vec4(0, 0, 0, 1));

    if (inTex->isSingleColor())
    {
        outputPins[0].propagateTexture(inTex);
        return;
    }

    Texture* outTex = nodeEvaluator->requestTexture(inTex->resolution);

    std::vector<PaintStroke> host_strokes;
    host_strokes.resize(1 << backupNumStrokes);
    const int numStrokes = host_strokes.size();

    glm::vec4* host_pixels = new glm::vec4[inTex->resolution.x * inTex->resolution.y];
    cudaMemcpy(host_pixels, inTex->dev_pixels, inTex->resolution.x * inTex->resolution.y * sizeof(glm::vec4), cudaMemcpyDeviceToHost);

    srand(1023940234);
    for (int i = 0; i < numStrokes; ++i)
    {
        glm::ivec2 pos(rand() % inTex->resolution.x, rand() % inTex->resolution.y);
        float scale = backupMinStrokeSize + (backupMaxStrokeSize - backupMinStrokeSize) * ((float)i / numStrokes);
        glm::vec4 color = host_pixels[pos.y * inTex->resolution.x + pos.x];

        host_strokes[i] = { 
            pos,
            scale,
            color 
        };
    }

    delete[] host_pixels;

    PaintStroke* dev_strokes;
    cudaMalloc(&dev_strokes, numStrokes * sizeof(PaintStroke));
    cudaMemcpy(dev_strokes, host_strokes.data(), numStrokes * sizeof(PaintStroke), cudaMemcpyHostToDevice);

    const dim3 blockSize(DEFAULT_BLOCK_SIZE_X, DEFAULT_BLOCK_SIZE_Y);
    const dim3 blocksPerGrid = calculateBlocksPerGrid(inTex->resolution, blockSize);

    kernPaint<<<blocksPerGrid, blockSize>>>(*outTex, dev_strokes, numStrokes);

    cudaFree(dev_strokes);

    outputPins[0].propagateTexture(outTex);
}

std::string NodePaintinator::debugGetSrcFileName() const
{
    return __FILE__;
}


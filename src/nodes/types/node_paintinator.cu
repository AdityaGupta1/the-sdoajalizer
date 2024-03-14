#include "node_paintinator.hpp"

#include "cuda_includes.hpp"

#include "random_utils.hpp"
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

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
        return NodeUI::IntEdit(backupMinStrokeSize, 0.15f, 1, backupMaxStrokeSize);
    case 3: // max stroke size
        ImGui::SameLine();
        return NodeUI::IntEdit(backupMaxStrokeSize, 0.15f, backupMinStrokeSize, 500);
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

struct PaintStrokeComparator
{
    __host__ __device__ bool operator()(const PaintStroke& stroke1, const PaintStroke& stroke2)
    {
        return stroke1.scale < stroke2.scale;
    }
};

__global__ void kernGeneratePaintStrokes(Texture inTex, PaintStroke* strokes, int numStrokes, int minStrokeSize, int maxStrokeSize)
{
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (idx >= numStrokes)
    {
        return;
    }

    thrust::default_random_engine rng = makeSeededRandomEngine(idx);
    thrust::uniform_int_distribution<int> distX(0, inTex.resolution.x - 1);
    thrust::uniform_int_distribution<int> distY(0, inTex.resolution.y - 1);
    glm::ivec2 pos(distX(rng), distY(rng));

    float scale = minStrokeSize + (maxStrokeSize - minStrokeSize) * ((float)idx / numStrokes);
    glm::vec4 color = inTex.dev_pixels[pos.y * inTex.resolution.x + pos.x];

    strokes[idx] = { glm::vec2(pos) + glm::vec2(0.5f), scale, color};
}

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

    PaintStroke* dev_strokes;
    const int numStrokes = 1 << backupNumStrokes;
    cudaMalloc(&dev_strokes, numStrokes * sizeof(PaintStroke));

    const dim3 blockSize1d(256);
    const dim3 blocksPerGrid1d(calculateNumBlocksPerGrid(numStrokes, blockSize1d.x));

    kernGeneratePaintStrokes<<<blocksPerGrid1d, blockSize1d>>>(*inTex, dev_strokes, numStrokes, backupMinStrokeSize, backupMaxStrokeSize);

    thrust::sort(thrust::device, dev_strokes, dev_strokes + numStrokes, PaintStrokeComparator());

    const dim3 blockSize2d(DEFAULT_BLOCK_SIZE_X, DEFAULT_BLOCK_SIZE_Y);
    const dim3 blocksPerGrid2d = calculateNumBlocksPerGrid(inTex->resolution, blockSize2d);

    kernPaint<<<blocksPerGrid2d, blockSize2d>>>(*outTex, dev_strokes, numStrokes);

    cudaFree(dev_strokes);

    outputPins[0].propagateTexture(outTex);
}

std::string NodePaintinator::debugGetSrcFileName() const
{
    return __FILE__;
}


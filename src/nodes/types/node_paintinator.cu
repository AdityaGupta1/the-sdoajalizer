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
    addPin(PinType::INPUT, "size bias").setNoConnect();

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
        return NodeUI::IntEdit(backupNumStrokes, 0.02f, 9, 19);
    case 2: // min stroke size
        ImGui::SameLine();
        return NodeUI::IntEdit(backupMinStrokeSize, 0.15f, 1, backupMaxStrokeSize);
    case 3: // max stroke size
        ImGui::SameLine();
        return NodeUI::IntEdit(backupMaxStrokeSize, 0.15f, backupMinStrokeSize, 500);
    case 4: // size bias
        ImGui::SameLine();
        return NodeUI::FloatEdit(backupSizeBias, 0.01f, 0.f, FLT_MAX);
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

__global__ void kernGeneratePaintStrokes(Texture inTex, PaintStroke* strokes, int numStrokes, int minStrokeSize, int maxStrokeSize, float sizeBias)
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

    thrust::uniform_real_distribution<float> u01(0, 1);
    float scale = minStrokeSize + (maxStrokeSize - minStrokeSize) * powf(u01(rng), sizeBias);
    glm::vec4 color = inTex.dev_pixels[pos.y * inTex.resolution.x + pos.x];

    strokes[idx] = { glm::vec2(pos) + glm::vec2(0.5f), scale, color};
}

#define NUM_SHARED_STROKES 512

__global__ void kernPaint(Texture outTex, PaintStroke* strokes, int numStrokes)
{
    __shared__ PaintStroke shared_strokes[NUM_SHARED_STROKES];
    __shared__ int shared_numFinishedThreads;

    const int localIdx = threadIdx.y * blockDim.x + threadIdx.x;

    if (localIdx == 0)
    {
        shared_numFinishedThreads = 0;
    }

    __syncthreads();

    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    const bool inBounds = x < outTex.resolution.x && y < outTex.resolution.y;

    if (!inBounds)
    {
        atomicAdd(&shared_numFinishedThreads, 1);
    }

    bool hasColor = false;
    glm::vec4 thisColor = glm::vec4(0, 0, 0, 0);
    glm::vec2 thisPos = glm::vec2(x, y);

    int strokesStart = 0;
    const int numTotalThreads = blockDim.x * blockDim.y;
    while (shared_numFinishedThreads != numTotalThreads && strokesStart < numStrokes)
    {
        if (localIdx < NUM_SHARED_STROKES)
        {
            // no issues with indices going out of bounds if numStrokes is a multiple of NUM_SHARED_STROKES
            shared_strokes[localIdx] = strokes[strokesStart + localIdx];
        }

        strokesStart += NUM_SHARED_STROKES;

        __syncthreads();

        if (inBounds && !hasColor)
        {
            for (int strokeIdx = 0; strokeIdx < NUM_SHARED_STROKES; ++strokeIdx)
            {
                const PaintStroke& stroke = shared_strokes[strokeIdx];

                if (glm::distance(thisPos, stroke.pos) <= stroke.scale)
                {
                    hasColor = true;
                    thisColor = stroke.color;
                    atomicAdd(&shared_numFinishedThreads, 1);
                    break;
                }
            }
        }

        __syncthreads();
    }

    if (!inBounds)
    {
        return;
    }

    const int idx = y * outTex.resolution.x + x;
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

    kernGeneratePaintStrokes<<<blocksPerGrid1d, blockSize1d>>>(*inTex, dev_strokes, numStrokes, backupMinStrokeSize, backupMaxStrokeSize, backupSizeBias);

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


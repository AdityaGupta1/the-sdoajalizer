#include "node_paintinator.hpp"

#include "cuda_includes.hpp"

#include "random_utils.hpp"
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <glm/gtx/component_wise.hpp>
#include <glm/gtc/constants.hpp>

#include "stb_image.h"

bool NodePaintinator::hasLoadedBrushes = false;
cudaArray_t NodePaintinator::pixelArray;
cudaTextureObject_t NodePaintinator::textureObject;

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

void NodePaintinator::freeDeviceMemory()
{
    cudaDestroyTextureObject(textureObject);
    cudaFreeArray(pixelArray);
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
        return NodeUI::FloatEdit(backupSizeBias, 0.01f, 0.01f, 100.f);
    default:
        throw std::runtime_error("invalid pin number");
    }
}

struct PaintStroke
{
    glm::vec2 pos;
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
    float sinVal, cosVal;
    sincosf(u01(rng) * glm::two_pi<float>(), &sinVal, &cosVal);
    glm::mat2 matRotate(cosVal, sinVal, -sinVal, cosVal);

    float scale = minStrokeSize + (maxStrokeSize - minStrokeSize) * powf(u01(rng), sizeBias);

    glm::vec3 color(inTex.dev_pixels[pos.y * inTex.resolution.x + pos.x]);

    thrust::uniform_int_distribution<int> distUv(0, 3);
    glm::vec2 uv(distUv(rng) * 0.25f, distUv(rng) * 0.25f);

    strokes[idx] = { glm::vec2(pos) + glm::vec2(0.5f), matRotate, scale, color, uv };
}

#define NUM_SHARED_STROKES 512

__global__ void kernPaint(Texture outTex, PaintStroke* strokes, int numStrokes, cudaTextureObject_t brushTex)
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
    glm::vec4 topColor = glm::vec4(0, 0, 0, 0);
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

                glm::vec2 localPos = stroke.matRotate * (thisPos - stroke.pos);
                if (glm::compMax(glm::abs(localPos)) > stroke.scale)
                {
                    continue;
                }

                glm::vec2 uv = ((localPos / stroke.scale) + 1.f) * 0.5f;

                uv = stroke.uv + uv * 0.25f;
                float4 bottomColor = tex2D<float4>(brushTex, uv.x, uv.y);
                if (bottomColor.w == 0.f)
                {
                    continue;
                }

                // probably not how real paint mixes but whatever
                glm::vec3 bottomRgb = glm::vec3(bottomColor.x, bottomColor.y, bottomColor.z) * stroke.color;
                float newAlpha = bottomColor.w + ((1.f - bottomColor.w) * topColor.a);
                topColor = glm::vec4(glm::mix(bottomRgb, glm::vec3(topColor), topColor.a), newAlpha);

                if (topColor.a > 0.999f)
                {
                    topColor.a = 1.f;
                    hasColor = true;
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
    outTex.dev_pixels[idx] = topColor;
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

    if (!hasLoadedBrushes)
    {
        int width, height, channels;
        unsigned char* host_pixels = stbi_load("assets/brushes/test.png", &width, &height, &channels, 4);

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
        int pitch = width * sizeof(uchar4);

        CUDA_CHECK(cudaMallocArray(
            &pixelArray,
            &channelDesc,
            width,
            height
        ));

        CUDA_CHECK(cudaMemcpy2DToArray(pixelArray,
            0, // wOffset
            0, // hOffset
            host_pixels,
            pitch,
            pitch,
            height,
            cudaMemcpyHostToDevice
        ));

        stbi_image_free(host_pixels);

        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = pixelArray;

        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeNormalizedFloat;
        texDesc.normalizedCoords = 1;
        texDesc.maxAnisotropy = 1;
        texDesc.maxMipmapLevelClamp = 99;
        texDesc.minMipmapLevelClamp = 0;
        texDesc.mipmapFilterMode = cudaFilterModePoint;
        texDesc.borderColor[0] = 1.0f;
        texDesc.sRGB = 0;

        CUDA_CHECK(cudaCreateTextureObject(&textureObject, &resDesc, &texDesc, nullptr));

        hasLoadedBrushes = true;
    }

    PaintStroke* dev_strokes;
    const int numStrokes = 1 << backupNumStrokes;
    CUDA_CHECK(cudaMalloc(&dev_strokes, numStrokes * sizeof(PaintStroke))); // TODO: malloc once and re-malloc only if numStrokes changes

    const dim3 blockSize1d(256);
    const dim3 blocksPerGrid1d(calculateNumBlocksPerGrid(numStrokes, blockSize1d.x));

    kernGeneratePaintStrokes<<<blocksPerGrid1d, blockSize1d>>>(
        *inTex,
        dev_strokes,
        numStrokes,
        backupMinStrokeSize,
        backupMaxStrokeSize,
        1.f / backupSizeBias
    );

    thrust::sort(thrust::device, dev_strokes, dev_strokes + numStrokes, PaintStrokeComparator());

    const dim3 blockSize2d(DEFAULT_BLOCK_SIZE_X, DEFAULT_BLOCK_SIZE_Y);
    const dim3 blocksPerGrid2d = calculateNumBlocksPerGrid(inTex->resolution, blockSize2d);

    kernPaint<<<blocksPerGrid2d, blockSize2d>>>(
        *outTex,
        dev_strokes,
        numStrokes,
        textureObject
    );

    CUDA_CHECK(cudaFree(dev_strokes));

    outputPins[0].propagateTexture(outTex);
}

std::string NodePaintinator::debugGetSrcFileName() const
{
    return __FILE__;
}


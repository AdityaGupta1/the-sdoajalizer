#include "node_paintinator.hpp"

#include "cuda_includes.hpp"

#include "random_utils.hpp"
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/shuffle.h>

#include <glm/gtx/component_wise.hpp>
#include <glm/gtc/constants.hpp>

#include "stb_image.h"

#include "npp_includes.hpp"

std::vector<BrushTexture> NodePaintinator::brushTextures = {
    { "assets/brushes/variety.png", "variety" },
    //{ "assets/brushes/debug_stars.png", "(DEBUG) stars" }
};

BrushTexture::BrushTexture(const std::string& filePath, const std::string& displayName)
    : filePath(filePath), displayName(displayName)
{}

void BrushTexture::load()
{
    int width, height, channels;
    unsigned char* host_pixels = stbi_load(filePath.c_str(), &width, &height, &channels, 4);

    scale = glm::vec2(width, height) / (float)std::max(width, height);

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

    CUDA_CHECK(cudaCreateTextureObject(&textureObj, &resDesc, &texDesc, nullptr));

    isLoaded = true;
}

NodePaintinator::NodePaintinator()
    : Node("paint-inator")
{
    addPin(PinType::OUTPUT, "image");

    addPin(PinType::INPUT, "image");
    addPin(PinType::INPUT, "brushes").setNoConnect();
    addPin(PinType::INPUT, "brush alpha").setNoConnect();
    addPin(PinType::INPUT, "min stroke size").setNoConnect();
    addPin(PinType::INPUT, "max stroke size").setNoConnect();
    addPin(PinType::INPUT, "grid size factor").setNoConnect();
    addPin(PinType::INPUT, "blur size factor").setNoConnect();
    addPin(PinType::INPUT, "new stroke threshold").setNoConnect();

    setExpensive();
}

NodePaintinator::~NodePaintinator()
{
    CUDA_CHECK(cudaFree(dev_strokes));
}

void NodePaintinator::freeDeviceMemory()
{
    for (const auto& brushTex : brushTextures)
    {
        if (brushTex.pixelArray != nullptr)
        {
            cudaDestroyTextureObject(brushTex.textureObj);
            cudaFreeArray(brushTex.pixelArray);
        }
    }
}

bool NodePaintinator::drawPinBeforeExtras(const Pin* pin, int pinNumber)
{
    if (pin->pinType == PinType::OUTPUT)
    {
        return false;
    }

    switch (pinNumber)
    {
    case 1: // brushes
        NodeUI::Separator("brushes");
        return false;
    case 3: // min stroke size
        NodeUI::Separator("stroke size");
        return false;
    case 5: // grid size factor
        NodeUI::Separator("stroke placement");
        return false;
    default:
        return false;
    }
}

static constexpr int minMinStrokeSize = 5;
static constexpr int maxMaxStrokeSize = 1000;

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
    case 1: // brushes
        ImGui::SameLine();
        return NodeUI::Dropdown<BrushTexture>(constParams.brushTexturePtr, brushTextures, [](const BrushTexture& brushTex) -> const char* {
            return brushTex.displayName.c_str();
        });
    case 2: // brush alpha
        ImGui::SameLine();
        return NodeUI::FloatEdit(constParams.brushAlpha, 0.01f, 0.f, 1.f);
    case 3: // min stroke size
        ImGui::SameLine();
        return NodeUI::IntEdit(constParams.minStrokeSize, 0.15f, minMinStrokeSize, constParams.maxStrokeSize);
    case 4: // max stroke size
        ImGui::SameLine();
        return NodeUI::IntEdit(constParams.maxStrokeSize, 0.15f, constParams.minStrokeSize, maxMaxStrokeSize);
    case 5: // grid size factor
        ImGui::SameLine();
        return NodeUI::FloatEdit(constParams.gridSizeFactor, 0.01f, 0.f, 1.f);
    case 6: // blur size factor
        ImGui::SameLine();
        return NodeUI::FloatEdit(constParams.blurKernelSizeFactor, 0.01f, 0.f, 3.f);
    case 7: // new stroke threshold
        ImGui::SameLine();
        return NodeUI::FloatEdit(constParams.newStrokeThreshold, 0.01f, 0.f, 3.f);
    default:
        throw std::runtime_error("invalid pin number");
    }
}

__global__ void kernFillEmptyTexture(Texture tex, int numPixels)
{
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (idx >= numPixels)
    {
        return;
    }

    tex.dev_pixels[idx] = glm::vec4(0, 0, 0, 0);
}

__global__ void kernCalculateColorDifference(Texture paintedTex, Texture refTex, float* colorDiff, int numPixels)
{
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (idx >= numPixels)
    {
        return;
    }

    glm::vec4 paintedCol = paintedTex.dev_pixels[idx];

    float diff;
    if (paintedCol.a == 0.f)
    {
        diff = 1e20f; // big number but not FLT_MAX to avoid overflow issues when summing error
    }
    else
    {
        diff = glm::distance(glm::vec3(paintedCol), glm::vec3(refTex.dev_pixels[idx]));
    }

    colorDiff[idx] = diff;
}

// I doubt this has coalesced memory accesses, which is probably not a good thing
__global__ void kernPrepareStrokes(Texture refTex, PaintStroke* strokes, int numStrokes)
{
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (idx >= numStrokes)
    {
        return;
    }

    PaintStroke& stroke = strokes[idx];

    auto rng = makeSeededRandomEngine(idx, numStrokes);
    thrust::uniform_real_distribution<float> u01(0, 1);
    float sinVal, cosVal;
    sincosf(u01(rng) * glm::two_pi<float>(), &sinVal, &cosVal);
    glm::mat2 matRotate = { cosVal, sinVal, -sinVal, cosVal };
    glm::mat2 matScale = glm::mat2(stroke.color.x, 0, 0, stroke.color.y);
    glm::mat3 matTranslate = { 1, 0, 0, 0, 1, 0, -stroke.pos.x, -stroke.pos.y, 1 };
    stroke.transform = glm::mat3(matScale * matRotate) * matTranslate;

    stroke.color = glm::vec3(refTex.dev_pixels[stroke.pos.y * refTex.resolution.x + stroke.pos.x]);

    thrust::uniform_int_distribution<int> distUv(0, 3);
    stroke.cornerUv = glm::vec2(distUv(rng), distUv(rng)) * 0.25f;
}

// probably not how real paint mixes but whatever
__device__ glm::vec4 blendPaintColors(const glm::vec4& bottomColor, const glm::vec4& topColor)
{
    float newAlpha = bottomColor.a + ((1.f - bottomColor.a) * topColor.a);
    return glm::vec4(glm::mix(glm::vec3(bottomColor), glm::vec3(topColor), topColor.a), newAlpha);
}

__device__ glm::vec2 matMul(const glm::mat3& mat, const glm::vec2 v)
{
    glm::vec3 mul = mat * glm::vec3(v, 1.f);
    return glm::vec2(mul) / mul.z;
}

#define NUM_SHARED_STROKES 512

__global__ void kernPaint(Texture outTex, PaintStroke* strokes, int numStrokes, cudaTextureObject_t brushTex, float brushAlpha)
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

    const bool inBounds = x < outTex.resolution.x&& y < outTex.resolution.y;

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
            int strokeIdx = strokesStart + localIdx;
            if (strokeIdx < numStrokes)
            {
                shared_strokes[localIdx] = strokes[strokesStart + localIdx];
            }
            else
            {
                shared_strokes[localIdx].pos.x = INT_MIN;
            }
        }

        strokesStart += NUM_SHARED_STROKES;

        __syncthreads();

        if (inBounds && !hasColor)
        {
            for (int strokeIdx = 0; strokeIdx < NUM_SHARED_STROKES; ++strokeIdx)
            {
                const PaintStroke& stroke = shared_strokes[strokeIdx];

                if (stroke.pos.x == INT_MIN)
                {
                    break;
                }

                glm::vec2 localPos = matMul(stroke.transform, thisPos);
                if (glm::compMax(glm::abs(localPos)) > 1.f)
                {
                    continue;
                }

                //glm::vec2 uv = (localPos + 1.f) * 0.5f;
                //uv = stroke.cornerUv + uv * 0.25f;
                glm::vec2 uv = stroke.cornerUv + (localPos + 1.f) * 0.125f;

                float4 brushColor = tex2D<float4>(brushTex, uv.x, uv.y);
                if (brushColor.w == 0.f)
                {
                    continue;
                }
                brushColor.w *= brushAlpha;

                glm::vec4 bottomColor(glm::vec3(brushColor.x, brushColor.y, brushColor.z) * stroke.color, brushColor.w);
                topColor = blendPaintColors(bottomColor, topColor);

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

    if (topColor.a == 0.f)
    {
        return;
    }

    const int idx = y * outTex.resolution.x + x;
    outTex.dev_pixels[idx] = blendPaintColors(outTex.dev_pixels[idx], topColor);
}

// TODO: make these into node parameters
static constexpr int numLayers = 7;

// reference paper: https://dl.acm.org/doi/10.1145/280814.280951
void NodePaintinator::evaluate()
{
    Texture* inTex = getPinTextureOrSingleColor(inputPins[0], glm::vec4(0, 0, 0, 1));

    if (inTex->isSingleColor())
    {
        outputPins[0].propagateTexture(inTex);
        return;
    }

    if (!constParams.brushTexturePtr->isLoaded)
    {
        constParams.brushTexturePtr->load();
    }

    Texture* outTex = nodeEvaluator->requestTexture(inTex->resolution);

    const int numPixels = outTex->resolution.x * outTex->resolution.y;
    const dim3 pixelsBlockSize1d(256);
    const dim3 pixelsBlocksPerGrid1d(calculateNumBlocksPerGrid(numPixels, pixelsBlockSize1d.x));

    kernFillEmptyTexture<<<pixelsBlocksPerGrid1d, pixelsBlockSize1d>>>(
        *outTex, numPixels
    );

    Texture* scratchTex = nodeEvaluator->requestTexture(inTex->resolution);
    Texture* refTex = nodeEvaluator->requestTexture(inTex->resolution);

    const int width = inTex->resolution.x;
    const int height = inTex->resolution.y;
    NppiSize oSrcSize = { width, height };
    NppiPoint oSrcOffset = { 0, 0 };

    NppiSize oSizeROI = { width, height };

    float* host_colorDiff = new float[numPixels];
    float* dev_colorDiff;
    CUDA_CHECK(cudaMalloc(&dev_colorDiff, numPixels * sizeof(float)));

    const dim3 blockSize2d(DEFAULT_BLOCK_SIZE_X, DEFAULT_BLOCK_SIZE_Y);
    const dim3 blocksPerGrid2d = calculateNumBlocksPerGrid(inTex->resolution, blockSize2d);

    float logMinStrokeSize = logf(constParams.minStrokeSize);
    float logMaxStrokeSize = logf(constParams.maxStrokeSize);
    for (int layerIdx = 0; layerIdx < numLayers; ++layerIdx)
    {
        // =========================
        // MAKE REFERENCE IMAGE
        // =========================

        float logStrokeSize = glm::mix(logMaxStrokeSize, logMinStrokeSize, (float)layerIdx / std::max(numLayers - 1, 1));
        float strokeSize = expf(logStrokeSize);

        const int kernelRadius = std::max((int)(strokeSize * constParams.blurKernelSizeFactor), 2); // radius < 2 leads to incorrect values (see https://www.desmos.com/calculator/jtsmwtzrc2)
        const int kernelDiameter = kernelRadius * 2 + 1;

        // TODO: malloc space for all kernels at once and fill them all using one kernel invocation
        //       also likely need to store all kernel sizes in a host vector
        //       this should significantly reduce the number of calls to cudaMalloc
        float* host_kernel = new float[kernelDiameter];
        float* dev_kernel;
        CUDA_CHECK(cudaMalloc(&dev_kernel, kernelDiameter * sizeof(float)));

        const float sigma = kernelDiameter / 9.f;
        const float sigma2 = sigma * sigma;
        const float normalizationFactor = 1.f / sqrtf(glm::two_pi<float>() * sigma2);
        const float exponentFactor = -0.5f / sigma2;

        for (int i = 0; i < kernelDiameter; ++i)
        {
            int x = i - kernelRadius;
            host_kernel[i] = normalizationFactor * expf(exponentFactor * x * x);
        }
        cudaMemcpy(dev_kernel, host_kernel, kernelDiameter * sizeof(float), cudaMemcpyHostToDevice);

        Npp32s nMaskSize = kernelDiameter;
        Npp32s nAnchor = kernelRadius;

        NPP_CHECK(nppiFilterColumnBorder_32f_C4R(
            (Npp32f*)inTex->dev_pixels, width * 4 * sizeof(float),
            oSrcSize, oSrcOffset,
            (Npp32f*)scratchTex->dev_pixels, width * 4 * sizeof(float),
            oSizeROI,
            (Npp32f*)dev_kernel, nMaskSize, nAnchor,
            NPP_BORDER_REPLICATE
        ));

        NPP_CHECK(nppiFilterRowBorder_32f_C4R(
            (Npp32f*)scratchTex->dev_pixels, width * 4 * sizeof(float),
            oSrcSize, oSrcOffset,
            (Npp32f*)refTex->dev_pixels, width * 4 * sizeof(float),
            oSizeROI,
            (Npp32f*)dev_kernel, nMaskSize, nAnchor,
            NPP_BORDER_REPLICATE
        ));

        delete[] host_kernel;
        CUDA_CHECK(cudaFree(dev_kernel));

        // =========================
        // PAINT LAYER
        // =========================

        kernCalculateColorDifference<<<pixelsBlocksPerGrid1d, pixelsBlockSize1d>>>(
            *outTex, *refTex, dev_colorDiff, numPixels
        );

        CUDA_CHECK(cudaMemcpy(host_colorDiff, dev_colorDiff, numPixels * sizeof(float), cudaMemcpyDeviceToHost));

        int gridSize = (int)(strokeSize * 2 * constParams.gridSizeFactor);
        gridSize = std::max(gridSize, 2);
        if (gridSize % 2 != 0) // ensure gridSize is even
        {
            --gridSize;
        }
        int halfGridSize = gridSize / 2;

        std::vector<PaintStroke> host_strokes;
        for (int cellY = 0; cellY < height + halfGridSize; cellY += gridSize)
        {
            for (int cellX = 0; cellX < width + halfGridSize; cellX += gridSize)
            {
                int xMin = std::max(cellX - halfGridSize, 0);
                int xMax = std::min(cellX + halfGridSize, width);
                int yMin = std::max(cellY - halfGridSize, 0);
                int yMax = std::min(cellY + halfGridSize, height);

                int numGridPixels = (xMax - xMin) * (yMax - yMin);

                // unsure if this is necessary
                if (numGridPixels == 0)
                {
                    continue;
                }

                float totalError = 0.f;
                float maxError = -FLT_MAX;
                glm::ivec2 maxErrorPos;
                for (int y = yMin; y < yMax; ++y)
                {
                    for (int x = xMin; x < xMax; ++x)
                    {
                        float error = host_colorDiff[y * width + x];
                        totalError += error;
                        if (error > maxError)
                        {
                            maxError = error;
                            maxErrorPos = glm::ivec2(x, y);
                        }
                    }
                }

                float areaError = totalError / numGridPixels;
                if (areaError < constParams.newStrokeThreshold)
                {
                    continue;
                }

                PaintStroke newStroke;
                newStroke.pos = maxErrorPos;
                newStroke.color.x = 1.f / (strokeSize * constParams.brushTexturePtr->scale.x);
                newStroke.color.y = 1.f / (strokeSize * constParams.brushTexturePtr->scale.y);
                // transform, color, and cornerUv are set by kernPrepareStrokes
                host_strokes.push_back(newStroke);
            }
        }

        const int numStrokes = host_strokes.size();
        if (numStrokes > numDevStrokes)
        {
            CUDA_CHECK(cudaFree(dev_strokes));
            CUDA_CHECK(cudaMalloc(&dev_strokes, numStrokes * sizeof(PaintStroke)));
        }

        CUDA_CHECK(cudaMemcpy(dev_strokes, host_strokes.data(), numStrokes * sizeof(PaintStroke), cudaMemcpyHostToDevice));

        auto rng = makeSeededRandomEngine(layerIdx, (int)strokeSize);
        thrust::shuffle(thrust::device, dev_strokes, dev_strokes + numStrokes, rng);

        const dim3 strokesBlockSize1d(256);
        const dim3 strokesBlocksPerGrid1d(calculateNumBlocksPerGrid(numStrokes, strokesBlockSize1d.x));
        kernPrepareStrokes<<<strokesBlocksPerGrid1d, strokesBlockSize1d>>>(
            *refTex, dev_strokes, numStrokes
        );

        kernPaint<<<blocksPerGrid2d, blockSize2d>>>(
            *outTex, dev_strokes, numStrokes, constParams.brushTexturePtr->textureObj, constParams.brushAlpha
        );
    }

    delete[] host_colorDiff;
    CUDA_CHECK(cudaFree(dev_colorDiff));

    outputPins[0].propagateTexture(outTex);
}

std::string NodePaintinator::debugGetSrcFileName() const
{
    return __FILE__;
}

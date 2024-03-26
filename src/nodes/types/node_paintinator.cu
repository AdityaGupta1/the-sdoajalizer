#include "node_paintinator.hpp"

#include "cuda_includes.hpp"

#include "random_utils.hpp"
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <glm/gtx/component_wise.hpp>
#include <glm/gtc/constants.hpp>

#include "stb_image.h"

#include "npp_includes.hpp"

bool NodePaintinator::hasLoadedBrushes = false;
cudaArray_t NodePaintinator::brushPixelArray;
cudaTextureObject_t NodePaintinator::brushTextureObj;

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
    cudaDestroyTextureObject(brushTextureObj);
    cudaFreeArray(brushPixelArray);
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
        return NodeUI::IntEdit(backupMinStrokeSize, 0.15f, 10, backupMaxStrokeSize);
    case 3: // max stroke size
        ImGui::SameLine();
        return NodeUI::IntEdit(backupMaxStrokeSize, 0.15f, backupMinStrokeSize, 1000);
    case 4: // size bias
        ImGui::SameLine();
        return NodeUI::FloatEdit(backupSizeBias, 0.01f, 0.01f, 100.f);
    default:
        throw std::runtime_error("invalid pin number");
    }
}

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

/*
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
*/

void NodePaintinator::loadBrushes()
{
    int width, height, channels;
    unsigned char* host_pixels = stbi_load("assets/brushes/test.png", &width, &height, &channels, 4);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    int pitch = width * sizeof(uchar4);

    CUDA_CHECK(cudaMallocArray(
        &brushPixelArray,
        &channelDesc,
        width,
        height
    ));

    CUDA_CHECK(cudaMemcpy2DToArray(brushPixelArray,
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
    resDesc.res.array.array = brushPixelArray;

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

    CUDA_CHECK(cudaCreateTextureObject(&brushTextureObj, &resDesc, &texDesc, nullptr));

    hasLoadedBrushes = true;
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

    thrust::default_random_engine rng = thrust::default_random_engine(hash(idx) ^ hash(numStrokes));
    thrust::uniform_real_distribution<float> u01(0, 1);
    float sinVal, cosVal;
    sincosf(u01(rng) * glm::two_pi<float>(), &sinVal, &cosVal);
    stroke.matRotate = { cosVal, sinVal, -sinVal, cosVal };

    stroke.color = glm::vec3(refTex.dev_pixels[stroke.pos.y * refTex.resolution.x + stroke.pos.x]);

    thrust::uniform_int_distribution<int> distUv(0, 3);
    stroke.uv = glm::vec2(distUv(rng) * 0.25f, distUv(rng) * 0.25f);
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

                glm::vec2 localPos = stroke.matRotate * (thisPos - glm::vec2(stroke.pos));
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
    // TODO: read existing color and blend accordingly
    if (topColor.a != 0.f)
    {
        outTex.dev_pixels[idx] = topColor;
    }
}

// TODO: make these into node parameters
static const int numLayers = 5;
static const float gridSizeFactor = 0.3f;
static const float newStrokeErrorThreshold = 0.3f;

// reference paper: https://dl.acm.org/doi/10.1145/280814.280951
void NodePaintinator::evaluate()
{
    Texture* inTex = getPinTextureOrSingleColor(inputPins[0], glm::vec4(0, 0, 0, 1));

    if (inTex->isSingleColor())
    {
        outputPins[0].propagateTexture(inTex);
        return;
    }

    if (!hasLoadedBrushes)
    {
        loadBrushes();
    }

    Texture* outTex = nodeEvaluator->requestTexture(inTex->resolution);

    const int numPixels = outTex->resolution.x * outTex->resolution.y;
    const dim3 blockSize1d(256);
    const dim3 blocksPerGrid1d(calculateNumBlocksPerGrid(numPixels, blockSize1d.x));

    kernFillEmptyTexture<<<blocksPerGrid1d, blockSize1d>>>(
        *outTex, numPixels
    );

    /*
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
        brushTextureObj
    );

    CUDA_CHECK(cudaFree(dev_strokes));

    outputPins[0].propagateTexture(outTex);
    */

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

    float logMinStrokeSize = logf(backupMinStrokeSize);
    float logMaxStrokeSize = logf(backupMaxStrokeSize);
    for (int i = 0; i < numLayers; ++i)
    {
        // =========================
        // MAKE REFERENCE IMAGE
        // =========================

        float logStrokeSize = glm::mix(logMaxStrokeSize, logMinStrokeSize, (float)i / std::max(numLayers - 1, 1));
        float strokeSize = expf(logStrokeSize);

        const int kernelRadius = (int)strokeSize; // TODO: check that this is correct and not off by a factor of 2
        const int kernelDiameter = kernelRadius * 2 + 1;

        // TODO: malloc space for all kernels at once and fill them all using one kernel invocation
        //       should significantly reduce the number of calls to cudaMalloc
        float* host_kernel = new float[kernelDiameter];
        float* dev_kernel;
        CUDA_CHECK(cudaMalloc(&dev_kernel, kernelDiameter * sizeof(float)));

        const float sigma = kernelDiameter / 9.f;
        const float sigma2 = sigma * sigma;
        const float normalizationFactor = 1.f / sqrtf(2 * glm::pi<float>() * sigma2);
        const float exponentFactor = -1.f / (2.f * sigma2);

        for (int i = 0; i < kernelDiameter; ++i)
        {
            int x = i - kernelRadius;
            host_kernel[i] = normalizationFactor * expf(exponentFactor * x * x);
        }
        cudaMemcpy(dev_kernel, host_kernel, kernelDiameter * sizeof(float), cudaMemcpyHostToDevice);

        Npp32s nMaskSize = kernelDiameter;
        Npp32s nAnchor = kernelRadius;

        NPP_CHECK(
            nppiFilterColumnBorder_32f_C4R(
                (Npp32f*)inTex->dev_pixels, width * 4 * sizeof(float),
                oSrcSize, oSrcOffset,
                (Npp32f*)scratchTex->dev_pixels, width * 4 * sizeof(float),
                oSizeROI,
                (Npp32f*)dev_kernel, nMaskSize, nAnchor,
                NPP_BORDER_REPLICATE)
        );

        NPP_CHECK(
            nppiFilterRowBorder_32f_C4R(
                (Npp32f*)scratchTex->dev_pixels, width * 4 * sizeof(float),
                oSrcSize, oSrcOffset,
                (Npp32f*)refTex->dev_pixels, width * 4 * sizeof(float),
                oSizeROI,
                (Npp32f*)dev_kernel, nMaskSize, nAnchor,
                NPP_BORDER_REPLICATE)
        );

        delete[] host_kernel;
        CUDA_CHECK(cudaFree(dev_kernel));

        // =========================
        // PAINT LAYER
        // =========================

        kernCalculateColorDifference<<<blocksPerGrid1d, blockSize1d>>>(
            *outTex, *refTex, dev_colorDiff, numPixels
        );

        CUDA_CHECK(cudaMemcpy(host_colorDiff, dev_colorDiff, numPixels * sizeof(float), cudaMemcpyDeviceToHost));

        // gridSize is always even and at least 2
        int gridSize = (int)(strokeSize * 2 * gridSizeFactor);
        if (gridSize % 2 != 0)
        {
            --gridSize;
        }
        gridSize = std::max(gridSize, 2);
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

                int gridPixels = (xMax - xMin) * (yMax - yMin);

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

                float areaError = totalError / gridPixels;
                if (areaError < newStrokeErrorThreshold)
                {
                    continue;
                }

                PaintStroke newStroke;
                newStroke.pos = maxErrorPos;
                newStroke.scale = strokeSize;
                // other fields are set by kernPrepareStrokes
                host_strokes.push_back(newStroke);
            }
        }

        // TODO: cudaMalloc dev_strokes only once based on maximum number of strokes for a layer
        PaintStroke* dev_strokes;
        const int numStrokes = host_strokes.size();
        CUDA_CHECK(cudaMalloc(&dev_strokes, numStrokes * sizeof(PaintStroke)));
        CUDA_CHECK(cudaMemcpy(dev_strokes, host_strokes.data(), numStrokes * sizeof(PaintStroke), cudaMemcpyHostToDevice));

        kernPrepareStrokes<<<blocksPerGrid1d, blockSize1d>>>(
            *refTex, dev_strokes, numStrokes
        );

        kernPaint<<<blocksPerGrid2d, blockSize2d>>>(
            *outTex, dev_strokes, numStrokes, brushTextureObj
        );

        CUDA_CHECK(cudaFree(dev_strokes));
    }

    delete[] host_colorDiff;
    CUDA_CHECK(cudaFree(dev_colorDiff));

    outputPins[0].propagateTexture(outTex);
}

std::string NodePaintinator::debugGetSrcFileName() const
{
    return __FILE__;
}


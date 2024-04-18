#include "node_lut.hpp"

#include "cuda_includes.hpp"

#include <fstream>
#include <sstream>
#include <iostream>

NodeLUT::NodeLUT()
    : Node("LUT")
{
    addPin(PinType::OUTPUT, "image");

    addPin(PinType::INPUT, "image");
    addPin(PinType::INPUT, "LUT").setNoConnect();
}

NodeLUT::~NodeLUT()
{
    freeTextureArray();
}

void NodeLUT::freeTextureArray()
{
    if (lutArray != nullptr)
    {
        cudaDestroyTextureObject(lutTexObj);
        cudaFreeArray(lutArray);

        lutArray = nullptr;
    }
}

// TODO: read from linear device memory containing LUT
__global__ void kernFillLut(cudaSurfaceObject_t surfObj, glm::vec3* lut, int lutSize)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int z = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (x >= lutSize || y >= lutSize || z >= lutSize)
    {
        return;
    }

    glm::vec3 lutEntry = lut[z * (lutSize * lutSize) + y * lutSize + x];
    float4 data = { lutEntry.r, lutEntry.g, lutEntry.b, 0.f };
    surf3Dwrite<float4>(data, surfObj, x * sizeof(float4), y, z);
}

void NodeLUT::reloadFile()
{
    freeTextureArray(); // TODO: keep array if LUT size is the same

    std::ifstream file(filePath);
    int lutSize = 0;

    if (!file.is_open())
    {
        std::cerr << "failed to open LUT file" << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (line.find("LUT_3D_SIZE") != std::string::npos)
        {
            std::istringstream iss(line);
            std::string temp;
            iss >> temp >> lutSize;
            break;
        }
    }

    if (lutSize == 0)
    {
        std::cerr << "LUT_3D_SIZE not found or invalid" << std::endl;
        return;
    }

    const int numEntries = lutSize * lutSize * lutSize;

    std::vector<glm::vec3> host_lut;
    host_lut.reserve(numEntries);

    float r, g, b;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        if (iss >> r >> g >> b)
        {
            host_lut.push_back(glm::vec3(r, g, b));
        }
    }

    file.close();

    assert(host_lut.size() == numEntries);

    glm::vec3* dev_lut;
    CUDA_CHECK(cudaMalloc(&dev_lut, numEntries * sizeof(glm::vec3))); // TODO: keep this memory malloc-ed and re-malloc only if size changes
    CUDA_CHECK(cudaMemcpy(dev_lut, host_lut.data(), numEntries * sizeof(glm::vec3), cudaMemcpyHostToDevice));

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>(); // float3 is not supported
    cudaExtent extent = { lutSize, lutSize, lutSize };

    CUDA_CHECK(cudaMalloc3DArray(
        &lutArray,
        &channelDesc,
        extent
    ));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = lutArray;

    cudaSurfaceObject_t surfObj;
    cudaCreateSurfaceObject(&surfObj, &resDesc);

    const dim3 blockSize3d(8, 8, 8);
    const dim3 blocksPerGrid3d = calculateNumBlocksPerGrid(glm::ivec3(lutSize), blockSize3d);
    kernFillLut<<<blocksPerGrid3d, blockSize3d>>>(surfObj, dev_lut, lutSize);  // TODO: pass in linear device memory containing LUT

    cudaFree(dev_lut);
    cudaDestroySurfaceObject(surfObj);

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;
    texDesc.maxAnisotropy = 1;
    texDesc.maxMipmapLevelClamp = 99;
    texDesc.minMipmapLevelClamp = 0;
    texDesc.mipmapFilterMode = cudaFilterModeLinear;
    texDesc.sRGB = 0;

    CUDA_CHECK(cudaCreateTextureObject(&lutTexObj, &resDesc, &texDesc, nullptr));
}

bool NodeLUT::drawPinExtras(const Pin* pin, int pinNumber)
{
    if (pin->pinType == PinType::OUTPUT)
    {
        return false;
    }

    ImGui::SameLine();

    bool didParameterChange;
    switch (pinNumber)
    {
    case 0: // image
        return false;
    case 1: // LUT
        didParameterChange = NodeUI::FilePicker(&filePath, { "Cube LUTs (.cube)", "*.cube" });
        break;
    default:
        throw std::runtime_error("invalid pin number");
    }

    if (didParameterChange)
    {
        needsReloadFile = true;
    }
    return didParameterChange;
}

__global__ void kernApplyLUT(Texture inTex, Texture outTex, cudaTextureObject_t lutTex)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= outTex.resolution.x && y >= outTex.resolution.y)
    {
        return;
    }

    const int idx = y * outTex.resolution.x + x;
    glm::vec4 inColLinear = inTex.getColor<TextureType::MULTI>(idx);

    glm::vec3 inColSrgb = ColorUtils::linearToSrgb(glm::vec3(inColLinear));
    float4 lutColSrgb = tex3D<float4>(lutTex, inColSrgb.x, inColSrgb.y, inColSrgb.z);
    glm::vec3 outColLinear = ColorUtils::srgbToLinear(glm::vec3(lutColSrgb.x, lutColSrgb.y, lutColSrgb.z));

    outTex.setColor<TextureType::MULTI>(idx, glm::vec4(outColLinear, inColLinear.a));
}

void NodeLUT::_evaluate()
{
    if (needsReloadFile)
    {
        reloadFile();
        needsReloadFile = false;
    }

    Texture* inTex = getPinTextureOrUniformColor(inputPins[0], glm::vec4(0, 0, 0, 1));

    // TODO: make this node work properly for uniform textures?
    //       probably unimportant since there's no good reason to apply a LUT to a single color
    if (inTex->isUniform() || lutArray == nullptr)
    {
        outputPins[0].propagateTexture(inTex);
        return;
    }

    Texture* outTex = nodeEvaluator->requestTexture<TextureType::MULTI>(inTex->resolution);

    const dim3 blockSize2d(DEFAULT_BLOCK_SIZE_X, DEFAULT_BLOCK_SIZE_Y);
    const dim3 blocksPerGrid2d = calculateNumBlocksPerGrid(inTex->resolution, blockSize2d);

    kernApplyLUT<<<blocksPerGrid2d, blockSize2d>>>(
        *inTex, *outTex, lutTexObj
    );

    outputPins[0].propagateTexture(outTex);
}

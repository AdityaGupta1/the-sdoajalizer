#include "node_separatecomponents.hpp"

#include "cuda_includes.hpp"

template<ComponentsType componentsType>
NodeSeparateComponents<componentsType>::NodeSeparateComponents(const std::string& name)
    : Node(name)
{
    if constexpr (componentsType == ComponentsType::RGB)
    {
        addPin(PinType::OUTPUT, "R").setSingleChannel();
        addPin(PinType::OUTPUT, "G").setSingleChannel();
        addPin(PinType::OUTPUT, "B").setSingleChannel();
    }
    else
    {
        addPin(PinType::OUTPUT, "H").setSingleChannel();
        addPin(PinType::OUTPUT, "S").setSingleChannel();
        addPin(PinType::OUTPUT, "V").setSingleChannel();
    }

    addPin(PinType::INPUT, "image");
}

template<ComponentsType componentsType>
bool NodeSeparateComponents<componentsType>::drawPinExtras(const Pin* pin, int pinNumber)
{
    if (pin->pinType == PinType::OUTPUT || pin->hasEdge())
    {
        return false;
    }

    switch (pinNumber)
    {
    case 0: // in color
        ImGui::SameLine();
        return NodeUI::ColorEdit4(constParams.color);
    default:
        throw std::runtime_error("invalid pin number");
    }
}

template<ComponentsType componentsType>
__host__ __device__ glm::vec3 separateComponents(glm::vec3 color)
{
    if constexpr (componentsType == ComponentsType::RGB)
    {
        return color;
    }
    else
    {
        return ColorUtils::rgbToHsv(color);
    }
}

template<ComponentsType componentsType>
__global__ void kernSeparateComponents(Texture inTex, Texture outTex1, Texture outTex2, Texture outTex3)
{
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (idx >= inTex.getNumPixels())
    {
        return;
    }

    glm::vec3 inColor = glm::vec3(inTex.getColor<TextureType::MULTI>(idx));
    glm::vec3 components = separateComponents<componentsType>(inColor);

    Texture* outTextures[3] = { &outTex1, &outTex2, &outTex3 };

    for (int compIdx = 0; compIdx < 3; ++compIdx)
    {
        Texture* outTex = outTextures[compIdx];
        if (outTex->getDevPixels<TextureType::SINGLE>() == nullptr)
        {
            continue;
        }

        outTex->setColor<TextureType::SINGLE>(idx, components[compIdx]);
    }
}

template<ComponentsType componentsType>
void NodeSeparateComponents<componentsType>::_evaluate()
{
    Texture* inTex = getPinTextureOrUniformColor(inputPins[0], ColorUtils::srgbToLinear(constParams.color));

    if (inTex->isUniform())
    {
        glm::vec3 inColor = glm::vec3(inTex->getUniformColor<TextureType::MULTI>());
        glm::vec3 components = separateComponents<componentsType>(inColor);

        for (int compIdx = 0; compIdx < 3; ++compIdx)
        {
            Pin& outputPin = outputPins[compIdx];
            if (!outputPin.hasEdge())
            {
                continue;
            }

            Texture* outTex = nodeEvaluator->requestUniformTexture();
            outTex->setUniformColor(components[compIdx]);
            outputPin.propagateTexture(outTex);
        }

        return;
    }

    Texture emptyTexture = Texture();
    Texture* outTextures[3];
    for (int compIdx = 0; compIdx < 3; ++compIdx)
    {
        Pin& outputPin = outputPins[compIdx];
        if (outputPin.hasEdge())
        {
            Texture* outTex = nodeEvaluator->requestTexture<TextureType::SINGLE>(inTex->resolution);
            outTextures[compIdx] = outTex;
        }
        else
        {
            outTextures[compIdx] = &emptyTexture;
        }
    }

    const dim3 blockSize(DEFAULT_BLOCK_SIZE_1D);
    const dim3 blocksPerGrid = calculateNumBlocksPerGrid(inTex->getNumPixels(), blockSize);
    kernSeparateComponents<componentsType><<<blocksPerGrid, blockSize>>>(
        *inTex,
        *outTextures[0], *outTextures[1], *outTextures[2]
    );

    for (int compIdx = 0; compIdx < 3; ++compIdx)
    {
        Pin& outputPin = outputPins[compIdx];
        if (outputPin.hasEdge())
        {
            outputPin.propagateTexture(outTextures[compIdx]);
        }
    }
}

template class NodeSeparateComponents<ComponentsType::RGB>;
template class NodeSeparateComponents<ComponentsType::HSV>;

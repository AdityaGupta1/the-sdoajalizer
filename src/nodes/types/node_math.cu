#include "node_math.hpp"

#include "cuda_includes.hpp"

std::vector<OperationName> NodeMath::operationNames = {
    { Operation::ADD, "add" },
    { Operation::SUBTRACT, "subtract" },
    { Operation::MULTIPLY, "multiply" },
    { Operation::DIVIDE, "divide" },
    { Operation::POWER, "power" },
    { Operation::MAX, "max" },
    { Operation::MIN, "min" }
};

NodeMath::NodeMath()
    : Node("math")
{
    addPin(PinType::OUTPUT, "output").setSingleChannel();

    addPin(PinType::INPUT, "input a").setSingleChannel();
    addPin(PinType::INPUT, "input b").setSingleChannel();
}

bool NodeMath::drawPinBeforeExtras(const Pin* pin, int pinNumber)
{
    if (pin->pinType == PinType::OUTPUT)
    {
        return false;
    }

    switch (pinNumber)
    {
    case 0: // input A
        return NodeUI::Dropdown<OperationName>(constParams.operationNamePtr, operationNames, [](const OperationName& operationName) -> const char* {
            return operationName.name.c_str();
        });
    default:
        return false;
    }
}

bool NodeMath::drawPinExtras(const Pin* pin, int pinNumber)
{
    if (pin->pinType == PinType::OUTPUT || pin->hasEdge())
    {
        return false;
    }

    switch (pinNumber)
    {
    case 0: // input a
        ImGui::SameLine();
        return NodeUI::FloatEdit(constParams.inputA, 0.01f);
    case 1: // input b
        ImGui::SameLine();
        return NodeUI::FloatEdit(constParams.inputB, 0.01f);
    default:
        throw std::runtime_error("invalid pin number");
    }
}

__host__ __device__ float performOperation(float inputA, float inputB, Operation operation)
{
    switch (operation)
    {
    case Operation::ADD:
        return inputA + inputB;
    case Operation::SUBTRACT:
        return inputA - inputB;
    case Operation::MULTIPLY:
        return inputA * inputB;
    case Operation::DIVIDE:
        if (inputB == 0.f)
        {
            return 0.f;
        }
        else
        {
            return inputA / inputB;
        }
    case Operation::POWER:
        return powf(inputA, inputB);
    case Operation::MAX:
        return fmaxf(inputA, inputB);
    case Operation::MIN:
        return fminf(inputA, inputB);
    }
}

__global__ void kernPerformOperation(Texture inTexA, Texture inTexB, Operation operation, Texture outTex)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= outTex.resolution.x || y >= outTex.resolution.y)
    {
        return;
    }

    float inputA = inTexA.getColorClamp<TextureType::SINGLE>(x, y);
    float inputB = inTexB.getColorClamp<TextureType::SINGLE>(x, y);

    outTex.setColor<TextureType::SINGLE>(x, y, performOperation(inputA, inputB, operation));
}

void NodeMath::_evaluate()
{
    Texture* inTexA = getPinTextureOrUniformColor(inputPins[0], constParams.inputA);
    Texture* inTexB = getPinTextureOrUniformColor(inputPins[1], constParams.inputB);

    Operation operation = constParams.operationNamePtr->operation;

    if (inTexA->isUniform() && inTexB->isUniform())
    {
        float inputA = inTexA->getUniformColor<TextureType::SINGLE>();
        float inputB = inTexB->getUniformColor<TextureType::SINGLE>();
        float result = performOperation(inputA, inputB, operation);

        Texture* outTex = nodeEvaluator->requestUniformTexture();
        outTex->setUniformColor(result);
        outputPins[0].propagateTexture(outTex);
        return;
    }

    glm::ivec2 outRes;
    if (inTexA->isUniform())
    {
        outRes = inTexB->resolution;
    }
    else
    {
        outRes = inTexA->resolution;
    }

    Texture* outTex = nodeEvaluator->requestTexture<TextureType::SINGLE>(outRes);

    const dim3 blockSize(DEFAULT_BLOCK_SIZE_2D_X, DEFAULT_BLOCK_SIZE_2D_Y);
    const dim3 blocksPerGrid = calculateNumBlocksPerGrid(outRes, blockSize);
    kernPerformOperation<<<blocksPerGrid, blockSize>>>(*inTexA, *inTexB, operation, *outTex);

    outputPins[0].propagateTexture(outTex);
}

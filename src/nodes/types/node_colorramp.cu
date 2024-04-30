#include "node_colorramp.hpp"

#include "cuda_includes.hpp"

NodeColorRamp::NodeColorRamp()
    : Node("color ramp")
{
    addPin(PinType::OUTPUT, "color");

    addPin(PinType::INPUT, "factor").setSingleChannel();
}

bool NodeColorRamp::drawPinBeforeExtras(const Pin* pin, int pinNumber)
{
    if (pin->pinType == PinType::OUTPUT || pin->hasEdge() || pinNumber != 0) // factor
    {
        return false;
    }

    ImGG::Settings settings{};
    settings.flags = ImGG::Flag::NoLabel | ImGG::Flag::NoDragDownToDelete | ImGG::Flag::NoBorder;
    settings.color_edit_flags = NodeUI::colorEditFlags;
    
    ImGui::Dummy({ settings.gradient_width + 20, 0.5f });
    return gradientWidget.widget("", settings);
}

bool NodeColorRamp::drawPinExtras(const Pin* pin, int pinNumber)
{
    if (pin->pinType == PinType::OUTPUT || pin->hasEdge())
    {
        return false;
    }

    switch (pinNumber)
    {
    case 0: // factor
        ImGui::SameLine();
        return NodeUI::FloatEdit(constParams.factor, 0.01f, 0.f, 1.f);
    default:
        throw std::runtime_error("invalid pin number");
    }
}

__host__ __device__ static glm::vec4 getRampColor(float pos, const ImGG::RawMark* marksStart, int numMarks, ImGG::Interpolation interpolationMode)
{
    pos = glm::clamp(pos, 0.f, 1.f);

    const ImGG::RawMark* marksEnd = marksStart + numMarks;
    const ImGG::RawMark* upper{ nullptr };
    const ImGG::RawMark* lower{ nullptr };
    while (marksStart < marksEnd)
    {
        if (marksStart->pos >= pos && (!upper || marksStart->pos < upper->pos))
        {
            upper = marksStart;
        }
        if (marksStart->pos <= pos && (!lower || marksStart->pos > lower->pos))
        {
            lower = marksStart;
        }

        marksStart++;
    }

    if (!lower && !upper)
    {
        return { 0.f, 0.f, 0.f, 1.f };
    }
    else if (upper && !lower)
    {
        return upper->color;
    }
    else if (!upper && lower)
    {
        return lower->color;
    }
    else if (upper == lower)
    {
        return upper->color;
    }

    return ImGG::rampInterpolate(lower, upper, pos, interpolationMode);
}

void NodeColorRamp::_evaluate()
{
    const auto& gradient = gradientWidget.gradient();
    const auto& marks = gradient.get_marks();

    if (marks.size() == 0)
    {
        Texture* outTex = nodeEvaluator->requestUniformTexture();
        outTex->setUniformColor(glm::vec4(0, 0, 0, 1));
        outputPins[0].propagateTexture(outTex);
        return;
    }

    std::vector<ImGG::RawMark> rawMarks;
    for (const auto& mark : marks)
    {
        float pos = mark.position.get();
        glm::vec4 color = ColorUtils::srgbToLinear(glm::vec4(mark.color.x, mark.color.y, mark.color.z, mark.color.w));
        rawMarks.emplace_back(pos, color);
    }

    Texture* inTex = getPinTextureOrUniformColor(inputPins[0], constParams.factor);

    if (inTex->isUniform())
    {
        Texture* outTex = nodeEvaluator->requestUniformTexture();
        outTex->setUniformColor(getRampColor(inTex->getUniformColor<TextureType::SINGLE>(), rawMarks.data(), rawMarks.size(), gradient.interpolation_mode()));
        outputPins[0].propagateTexture(outTex);
        return;
    }

    //Texture* outTex = nodeEvaluator->requestTexture<TextureType::MULTI>(inTex->resolution);

    //const dim3 blockSize(DEFAULT_BLOCK_SIZE_1D);
    //const dim3 blocksPerGrid = calculateNumBlocksPerGrid(inTex->getNumPixels(), blockSize);
    //kernInvert<<<blocksPerGrid, blockSize>>>(*inTex, *outTex);

    //outputPins[0].propagateTexture(outTex);
}

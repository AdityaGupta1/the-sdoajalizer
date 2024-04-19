#include "node.hpp"

#include "ImGui/imnodes.h"

#define DEBUG_SHOW_NODE_ID 0

int Node::nextId = 0;

Node::Node(std::string name)
    : name(name), id(Node::nextId)
{
    Node::nextId += NODE_ID_STRIDE;
}

Node::~Node()
{}

Pin& Node::addPin(PinType type, const std::string& name)
{
    int pinId = this->id + inputPins.size() + outputPins.size() + 1;
    auto& pinVector = (type == PinType::INPUT) ? inputPins : outputPins;
    return pinVector.emplace_back(pinId, this, type, name);
}

Pin& Node::addPin(PinType type)
{
    return addPin(type, type == PinType::INPUT ? "input" : "output");
}

bool Node::getIsExpensive()
{
    return this->isExpensive;
}

void Node::setExpensive()
{
    this->isExpensive = true;
}

unsigned int Node::getTitleBarColor() const
{
    return IM_COL32(11, 109, 191, 255);
}

unsigned int Node::getTitleBarHoveredColor() const
{
    return IM_COL32(81, 148, 204, 255);
}

Texture* Node::getPinTextureOrUniformColor(const Pin& pin, glm::vec4 col)
{
    Texture* tex = pin.getSingleTexture();

    if (tex == nullptr)
    {
        tex = nodeEvaluator->requestUniformTexture();
        tex->setUniformColor(col);
    }

    return tex;
}

Texture* Node::getPinTextureOrUniformColor(const Pin& pin, float col)
{
    return getPinTextureOrUniformColor(pin, Texture::singleToMulti(col));
}

// can potentially add pre- and post-effects to this function
void Node::evaluate()
{
    _evaluate();
}

void Node::clearInputTextures()
{
    for (auto& inputPin : this->inputPins)
    {
        inputPin.clearTextures();
    }
}

bool Node::getIsBeingEvaluated()
{
    return this->isBeingEvaluated;
}

void Node::setIsBeingEvaluated(bool isBeingEvaluated)
{
    this->isBeingEvaluated = isBeingEvaluated;
}

Pin& Node::getPin(int pinId)
{
    for (auto& inputPin : inputPins) 
    {
        if (inputPin.id == pinId) {
            return inputPin;
        }
    }

    for (auto& outputPin : outputPins) {
        if (outputPin.id == pinId) {
            return outputPin;
        }
    }

    throw std::runtime_error("invalid pin id");
}

void Node::setNodeEvaluator(NodeEvaluator* nodeEvaluator)
{
    this->nodeEvaluator = nodeEvaluator;
}

void Node::drawPin(const Pin& pin, int pinNumber, bool& didParameterChange)
{
    ImNodes::PushColorStyle(ImNodesCol_Pin, pin.getColor());
    ImNodes::PushColorStyle(ImNodesCol_PinHovered, pin.getHoveredColor());

    didParameterChange |= drawPinBeforeExtras(&pin, pinNumber);

    if (pin.getCanConnect())
    {
        if (pin.pinType == PinType::INPUT)
        {
            ImNodes::BeginInputAttribute(pin.id);
        }
        else
        {
            ImNodes::BeginOutputAttribute(pin.id);
        }
    }

    ImGui::Text(pin.name.c_str());
    didParameterChange |= drawPinExtras(&pin, pinNumber);

    if (pin.getCanConnect())
    {
        if (pin.pinType == PinType::INPUT)
        {
            ImNodes::EndInputAttribute();
        }
        else
        {
            ImNodes::EndOutputAttribute();
        }
    }

    didParameterChange |= drawPinAfterExtras(&pin, pinNumber);

    ImNodes::PopColorStyle();
    ImNodes::PopColorStyle();
}

bool Node::draw()
{
    ImNodes::PushColorStyle(ImNodesCol_TitleBar, this->getTitleBarColor());
    ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered, this->getTitleBarHoveredColor());
    ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, this->getTitleBarHoveredColor());

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(5.f, 5.f)); // for padding inside popups

    ImNodes::BeginNode(this->id);

    ImNodes::BeginNodeTitleBar();
#if DEBUG_SHOW_NODE_ID
    ImGui::TextUnformatted((this->name + " (" + std::to_string(this->id) + ")").c_str());
#else
    ImGui::TextUnformatted(this->name.c_str());
#endif
    ImNodes::EndNodeTitleBar();

    bool didParameterChange = false;

    for (int i = 0; i < outputPins.size(); ++i)
    {
        drawPin(outputPins[i], i, didParameterChange);
    }

    for (int i = 0; i < inputPins.size(); ++i)
    {
        drawPin(inputPins[i], i, didParameterChange);
    }

    ImNodes::EndNode();

    ImGui::PopStyleVar();

    ImNodes::PopColorStyle();
    ImNodes::PopColorStyle();
    ImNodes::PopColorStyle();

    return didParameterChange;
}

bool Node::drawPinBeforeExtras(const Pin* pin, int pinNumber)
{
    // do nothing, should be overridden by nodes with parameters
    return false;
}

bool Node::drawPinExtras(const Pin* pin, int pinNumber)
{
    // do nothing, should be overridden by nodes with parameters
    return false;
}

bool Node::drawPinAfterExtras(const Pin* pin, int pinNumber)
{
    // do nothing, should be overridden by nodes with parameters
    return false;
}

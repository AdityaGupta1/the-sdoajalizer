#include "node.hpp"

#include "ImGui/imnodes.h"

#define DEBUG_SHOW_NODE_ID 0

Pin::Pin(int id, Node* node, PinType pinType, const std::string& name)
    : id(id), node(node), pinType(pinType), name(name)
{}

Node* Pin::getNode() const
{
    return this->node;
}

const std::unordered_set<Edge*>& Pin::getEdges() const
{
    return this->edges;
}

bool Pin::hasEdge() const
{
    return !getEdges().empty();
}

void Pin::addEdge(Edge* edge)
{
    this->edges.insert(edge);
}

void Pin::removeEdge(Edge* edge)
{
    this->edges.erase(edge);
}

void Pin::clearEdges()
{
    this->edges.clear();
}

Texture* Pin::getSingleTexture() const
{
    if (this->edges.empty())
    {
        return nullptr;
    }

    return (*edges.begin())->getTexture();
}

void Pin::propagateTexture(Texture* texture)
{
    for (auto& edge : this->edges)
    {
        edge->setTexture(texture);
    }

    if (this->cachedTexture != nullptr)
    {
        printf("WARNING: calling propagateTexture() when cachedTexture != nullptr\n");
        --this->cachedTexture->numReferences;
        this->cachedTexture = nullptr;
    }

    if (this->cacheState == PinCacheState::PREPARED)
    {
        this->cachedTexture = texture;
        ++this->cachedTexture->numReferences;
        this->cacheState = PinCacheState::CACHED;
    }
}

void Pin::clearTextures()
{
    for (auto& edge : this->edges)
    {
        edge->clearTexture();
    }
}

void Pin::setNoConnect()
{
    this->canConnect = false;
}

bool Pin::getCanConnect() const
{
    return this->canConnect;
}

PinCacheState Pin::getCacheState() const
{
    return this->cacheState;
}

Texture* Pin::getCachedTexture() const
{
    return this->cachedTexture;
}

void Pin::prepareForCache()
{
    if (this->cacheState != PinCacheState::CACHED)
    {
        this->cacheState = PinCacheState::PREPARED;
    }
}

void Pin::deleteCache()
{
    this->cacheState = PinCacheState::NO_CACHE;

    if (this->cachedTexture != nullptr)
    {
        --this->cachedTexture->numReferences;
        this->cachedTexture = nullptr;
    }
}

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

void Node::setExpensive()
{
    this->isExpensive = true;
}

unsigned int Node::getTitleBarColor() const
{
    return IM_COL32(11, 109, 191, 255);
}

unsigned int Node::getTitleBarSelectedColor() const
{
    return IM_COL32(81, 148, 204, 255);
}

Texture* Node::getPinTextureOrSingleColor(const Pin& pin, glm::vec4 col)
{
    Texture* tex = pin.getSingleTexture();

    if (tex == nullptr)
    {
        tex = nodeEvaluator->requestSingleColorTexture();
        tex->setSingleColor(col);
    }

    return tex;
}

Texture* Node::getPinTextureOrSingleColor(const Pin& pin, float col)
{
    return getPinTextureOrSingleColor(pin, glm::vec4(col, col, col, 1));
}

void Node::clearInputTextures()
{
    for (auto& inputPin : this->inputPins)
    {
        inputPin.clearTextures();
    }
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
}

bool Node::draw()
{
    ImNodes::PushColorStyle(ImNodesCol_TitleBar, this->getTitleBarColor());
    ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered, this->getTitleBarSelectedColor());
    ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, this->getTitleBarSelectedColor());

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

bool Node::drawPinExtras(const Pin* pin, int pinNumber)
{
    // do nothing, should be overridden by nodes with parameters
    return false;
}

void Node::debugOpenSrcFile()
{
    system(("\"" + debugGetSrcFileName() + "\"").c_str());
}

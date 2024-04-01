#pragma once

#define NODE_ID_STRIDE 32

#define DEFAULT_BLOCK_SIZE_X 32
#define DEFAULT_BLOCK_SIZE_Y 32

#include "node_evaluator.hpp"
#include "node_ui_elements.hpp"
#include "node_utils.hpp"
#include "texture.hpp"
#include "color_utils.hpp"

#include "ImGui/imgui.h"

#include <vector>
#include <string>
#include <unordered_set>
#include <stdexcept>

class Edge;
class Node;
class NodeEvaluator;

enum class PinType
{
    INPUT, OUTPUT
};

enum class PinCacheState
{
    NO_CACHE, PREPARED, CACHED
};

class Pin
{
private:
    Node* node{ nullptr };
    std::unordered_set<Edge*> edges;

    bool canConnect{ true };

    PinCacheState cacheState{ PinCacheState::NO_CACHE };
    Texture* cachedTexture{ nullptr };

public:
    const int id;
    const PinType pinType;
    const std::string name;

    Pin(int id, Node* node, PinType pinType, const std::string& name);

    Node* getNode() const;
    const std::unordered_set<Edge*>& getEdges() const;
    bool hasEdge() const;

    void addEdge(Edge* edge);
    void removeEdge(Edge* edge);
    void clearEdges();

    // utility function to get single texture for input pins (nullptr if no connected edge)
    Texture* getSingleTexture() const;

    void propagateTexture(Texture* texture);
    void clearTextures();

    void setNoConnect();
    bool getCanConnect() const;

    PinCacheState getCacheState() const;
    Texture* getCachedTexture() const;
    void prepareForCache();
    void deleteCache();
};

class Node
{
    friend class NodeEvaluator;

private:
    static int nextId;

    bool isExpensive{ false };

protected:
    const std::string name;

    NodeEvaluator* nodeEvaluator{ nullptr };

    Node(std::string name);

public:
    virtual ~Node();

protected:
    Pin& addPin(PinType type, const std::string& name);
    Pin& addPin(PinType type);

    void setExpensive();

    virtual unsigned int getTitleBarColor() const;
    virtual unsigned int getTitleBarSelectedColor() const;

    virtual void evaluate() = 0;
    Texture* getPinTextureOrSingleColor(const Pin& pin, glm::vec4 col);
    Texture* getPinTextureOrSingleColor(const Pin& pin, float col);
    void clearInputTextures();

    virtual bool drawPinBeforeExtras(const Pin* pin, int pinNumber);
    virtual bool drawPinExtras(const Pin* pin, int pinNumber);
    virtual bool drawPinAfterExtras(const Pin* pin, int pinNumber);

    virtual std::string debugGetSrcFileName() const = 0;

public:
    const int id;

    std::vector<Pin> inputPins;
    std::vector<Pin> outputPins;

    Pin& getPin(int pinId);

    void setNodeEvaluator(NodeEvaluator* nodeEvaluator);

private:
    void drawPin(const Pin& pin, int pinNumber, bool& didParameterChange);

public:
    bool draw();

    // I would rather not implement this but it was suggested so I have no choice
    void debugOpenSrcFile();
};

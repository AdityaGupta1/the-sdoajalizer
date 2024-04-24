#pragma once

#define NODE_ID_STRIDE 32

#define DEFAULT_BLOCK_SIZE_1D 512
#define DEFAULT_BLOCK_SIZE_2D_X 32
#define DEFAULT_BLOCK_SIZE_2D_Y 32

#include "pin.hpp"
#include "node_enums.hpp"
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
class Pin;
class NodeEvaluator;

class Node
{
private:
    static int nextId;

    bool isExpensive{ false };

protected:
    const std::string name;

    NodeEvaluator* nodeEvaluator{ nullptr };
    bool isBeingEvaluated{ false };

    Node(const std::string& name);

public:
    virtual ~Node();

protected:
    Pin& addPin(PinType type, const std::string& name);
    Pin& addPin(PinType type);

    void setExpensive();

    virtual unsigned int getTitleBarColor() const;
    virtual unsigned int getTitleBarHoveredColor() const;

    virtual void _evaluate() = 0;
    Texture* getPinTextureOrUniformColor(const Pin& pin, glm::vec4 col);
    Texture* getPinTextureOrUniformColor(const Pin& pin, float col);

    virtual bool drawPinBeforeExtras(const Pin* pin, int pinNumber);
    virtual bool drawPinExtras(const Pin* pin, int pinNumber);
    virtual bool drawPinAfterExtras(const Pin* pin, int pinNumber);

public:
    const int id;

    std::vector<Pin> inputPins, outputPins;

    Pin& getPin(int pinId);

    void setNodeEvaluator(NodeEvaluator* nodeEvaluator);

    void evaluate();
    void clearInputTextures();

    bool getIsExpensive();

    bool getIsBeingEvaluated();
    void setIsBeingEvaluated(bool isBeingEvaluated);

private:
    void drawPin(const Pin& pin, int pinNumber, bool& didParameterChange);

public:
    bool draw();
};

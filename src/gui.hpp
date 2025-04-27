#pragma once

#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <unordered_map>
#include <memory>
#include <string>

#include "nodes/node.hpp"
#include "nodes/edge.hpp"
#include "nodes/node_evaluator.hpp"

class Gui
{
private:
    GLFWwindow* window;
    ImGuiIO* io{ nullptr };

    std::unordered_map<int, std::unique_ptr<Node>> nodes;
    std::unordered_map<int, std::unique_ptr<Edge>> edges;

    Node* outputNode;
    NodeEvaluator nodeEvaluator{ glm::ivec2(1920, 1080) * 2 }; // TODO: allow user to set this

    bool isFirstRender{ true };
    bool isNetworkDirty{ true };

    struct {
        bool deleteComponents{ false };
        bool shouldCreateWindowBeVisible{ false };
    } controls;

    struct {
        bool visible{ false };
        ImVec2 pos{ 0, 0 };

        bool justOpened{ false };

        int id{ 0 };
    } createWindowData;

    using NodeCreator = std::pair<std::string, std::function<std::unique_ptr<Node>()>>;
    std::vector<NodeCreator> nodeCreators;

public:
    void setupNodeCreators();

    void init(GLFWwindow* window);
    void deinit();

private:
    void setupStyle();

    Pin& getPin(int pinId);

    int addNode(std::unique_ptr<Node> node);
    void addEdge(int startPinId, int endPinId);

    void deleteNode(int nodeId);
    void deleteEdge(int edgeId);
    void deleteEdge(Edge* edge);
    void deletePinEdges(Pin& pin);

    void saveImage();

    void drawOutputImageViewer();
    void drawNodeEditor();
    void updateNodeCreatorWindow();

public:
    void render();

    void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
    void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
};

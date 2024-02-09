#pragma once

#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include <unordered_map>
#include <memory>

#include "nodes/node.hpp"
#include "nodes/edge.hpp"

class Gui
{
private:
    GLFWwindow* window;
    ImGuiIO* io{ nullptr };

    std::unordered_map<int, std::unique_ptr<Node>> nodes;
    std::unordered_map<int, std::unique_ptr<Edge>> edges;

    bool isDeleteQueued{ false };

public:
    void init(GLFWwindow* window);
    void deinit();

private:
    void setupStyle();

    Pin& getPin(int pinId);

    void addNode(std::unique_ptr<Node> node);
    void addEdge(int startPinId, int endPinId);

    void deleteNode(int nodeId);
    void deleteEdge(int edgeId);
    void deleteEdge(Edge* edge);
    void deletePinEdges(Pin& pin);

    void drawNodeEditor();

public:
    void render();

    void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
    void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
};

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
    ImGuiIO* io{ nullptr };

    std::unordered_map<int, std::unique_ptr<Node>> nodes;
    std::unordered_map<int, std::unique_ptr<Edge>> edges;

    Pin& getPin(int pinId);

    void addNode(std::unique_ptr<Node> node);
    void addEdge(int startPinId, int endPinId);

public:
    void init(GLFWwindow* window);
    void deinit();

    void setupStyle();

    void render();
};

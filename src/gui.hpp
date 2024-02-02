#pragma once

#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

class Gui
{
private:
    ImGuiIO* io{ nullptr };

public:
    void init(GLFWwindow* window);
    void deinit();

    void render();
};
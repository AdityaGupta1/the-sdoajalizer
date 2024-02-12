#include "node_ui_elements.hpp"

#include "ImGui/imgui.h"

bool NodeUI::ColorEdit4(glm::vec4& col)
{
    return ImGui::ColorEdit4("", glm::value_ptr(col), ImGuiColorEditFlags_NoInputs);
}
#pragma once

#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui.h>

namespace ImGG { namespace internal {

inline auto line_height() -> float
{
    return ImGui::GetFrameHeight();
}

inline auto button_size() -> ImVec2
{
    return ImVec2{line_height(), line_height()};
}

inline auto gradient_position(float x_offset) -> ImVec2
{
    ImVec2 cursorScreenPos = ImGui::GetCursorScreenPos();
    return ImVec2(cursorScreenPos.x + x_offset, cursorScreenPos.y);
}

inline auto border_color() -> ImU32
{
    return ImGui::GetColorU32(ImGuiCol_Border);
}

}} // namespace ImGG::internal
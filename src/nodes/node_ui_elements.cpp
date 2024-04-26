#include "node_ui_elements.hpp"

#include "ImGui/imgui_internal.h"

#include "portable_file_dialogs.h"

void NodeUI::Separator(const std::string& text)
{
    ImGui::Spacing();

    ImGui::Text(text.c_str());

    ImVec2 textSize = ImGui::CalcTextSize(text.c_str());
    ImVec2 lineStart = ImGui::GetCursorScreenPos();
    lineStart.y -= 4.f;
    ImVec2 lineEnd = ImVec2(lineStart.x + textSize.x, lineStart.y);

    ImGui::GetWindowDrawList()->AddLine(
        lineStart, lineEnd,
        ImGui::ColorConvertFloat4ToU32(ImGui::GetStyle().Colors[ImGuiCol_Text]),
        1.0f);
}

// somewhat hacky way to limit true values to using the slider or confirming after typing into temp input (i.e. not for each individual character typed into temp input)
bool changeGate(bool didParameterChange)
{
    bool tempActive = ImGui::TempInputIsOrWasActive(ImGui::GetItemID());
    return (didParameterChange && !tempActive) // slider
        || (!didParameterChange && tempActive && ImGui::IsItemDeactivatedAfterEdit()); // temp input
}

constexpr unsigned int colorEditFlags =
ImGuiColorEditFlags_NoOptions |
ImGuiColorEditFlags_NoInputs |
ImGuiColorEditFlags_AlphaBar |
ImGuiColorEditFlags_AlphaPreview |
ImGuiColorEditFlags_HDR |
ImGuiColorEditFlags_Float |
ImGuiColorEditFlags_PickerHueWheel;

bool NodeUI::ColorEdit4(glm::vec4& col)
{
    ImGui::PushID(&col);

    bool didParameterChange = changeGate(ImGui::ColorEdit4("", glm::value_ptr(col), colorEditFlags));

    ImGui::PopID();

    return didParameterChange;
}

bool NodeUI::FloatEdit(float& v, float v_speed, float v_min, float v_max, const char* format)
{
    ImGui::PushID(&v);
    ImGui::PushItemWidth(80);

    bool didParameterChange = changeGate(ImGui::DragFloat("", &v, v_speed, v_min, v_max, format, ImGuiSliderFlags_AlwaysClamp));

    ImGui::PopID();
    ImGui::PopItemWidth();

    return didParameterChange;
}

bool NodeUI::IntEdit(int& v, float v_speed, int v_min, int v_max, const char* format)
{
    ImGui::PushID(&v);
    ImGui::PushItemWidth(80);

    bool didParameterChange = changeGate(ImGui::DragInt("", &v, v_speed, v_min, v_max, format, ImGuiSliderFlags_AlwaysClamp));

    ImGui::PopID();
    ImGui::PopItemWidth();

    return didParameterChange;
}

bool NodeUI::Checkbox(bool& v, const std::string& label)
{
    ImGui::PushID(&v);

    bool didParameterChange = ImGui::Checkbox(label.c_str(), &v);

    ImGui::PopID();

    return didParameterChange;
}

bool NodeUI::FilePicker(std::string* filePath, const std::vector<std::string>& filters)
{
    ImGui::PushItemWidth(160);
    char* fileName = const_cast<char*>(filePath->c_str()) + filePath->find_last_of("/\\") + 1;
    // sussy const_cast but it should be fine since it's read-only
    ImGui::InputText("", fileName, filePath->length(), ImGuiInputTextFlags_ReadOnly);
    ImGui::PopItemWidth();

    ImGui::SameLine();
    if (!ImGui::Button("open")) {
        return false;
    }

    auto selections = pfd::open_file("Open", "", filters).result();
    if (selections.empty()) {
        return false;
    }

    const std::string& newFilePath = selections[0];
    if (*filePath == newFilePath) {
        return false;
    }

    *filePath = newFilePath;
    return true;
}

bool NodeUI::Dropdown(int& selectedItem, const std::vector<const char*>& items)
{
    ImGui::PushID(&selectedItem);
    ImGui::PushItemWidth(120);

    bool didParameterChange = ImGui::Combo("", &selectedItem, items.data(), items.size());

    ImGui::PopID();
    ImGui::PopItemWidth();

    return didParameterChange;
}

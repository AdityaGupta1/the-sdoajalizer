#include "node_ui_elements.hpp"

#include "ImGui/imgui.h"

#include "portable_file_dialogs.h"

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

    bool didParameterChange = ImGui::ColorEdit4("", glm::value_ptr(col), colorEditFlags);

    ImGui::PopID();

    return didParameterChange;
}

bool NodeUI::FloatEdit(float& v, float v_speed, float v_min, float v_max, const char* format)
{
    ImGui::PushID(&v);
    ImGui::PushItemWidth(80);

    bool didParameterChange = ImGui::DragFloat("", &v, v_speed, v_min, v_max, format, ImGuiSliderFlags_AlwaysClamp);

    ImGui::PopID();
    ImGui::PopItemWidth();

    return didParameterChange;
}

bool NodeUI::IntEdit(int& v, float v_speed, int v_min, int v_max, const char* format)
{
    ImGui::PushID(&v);
    ImGui::PushItemWidth(80);

    bool didParameterChange = ImGui::DragInt("", &v, v_speed, v_min, v_max, format, ImGuiSliderFlags_AlwaysClamp);

    ImGui::PopID();
    ImGui::PopItemWidth();

    return didParameterChange;
}

bool NodeUI::FilePicker(std::string* filePath)
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

    auto selections = pfd::open_file("Open", "", { "Image Files (.png, .jpg, .jpeg, .exr)", "*.png *.jpg *.jpeg *.exr" }).result();
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

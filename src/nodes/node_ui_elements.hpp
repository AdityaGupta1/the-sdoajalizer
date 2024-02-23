#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <string>
#include <vector>

namespace NodeUI
{
    static const float defaultBackupFloat = 0.5f;
    static const glm::vec2 defaultBackupVec2 = { 0.5f, 0.5f };
    static const glm::vec3 defaultBackupVec3 = { 0.5f, 0.5f, 0.5f };
    static const glm::vec4 defaultBackupVec4 = { 0.5f, 0.5f, 0.5f, 1.f };

    bool ColorEdit4(glm::vec4& col);

    bool FloatEdit(float& v, float v_speed = 1.0f, float v_min = 0.0f, float v_max = 0.0f, const char* format = "%.3f");
    bool IntEdit(int& v, float v_speed = 1.0f, int v_min = 0, int v_max = 0, const char* format = "%d");

    bool FilePicker(std::string* filePath);

    bool Dropdown(int& selectedItem, const std::vector<const char*>& items);
}

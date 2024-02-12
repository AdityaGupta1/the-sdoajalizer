#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace NodeUI
{
    static const float defaultBackupFloat = 0.5f;
    static const glm::vec2 defaultBackupVec2 = { 0.5f, 0.5f };
    static const glm::vec3 defaultBackupVec3 = { 0.5f, 0.5f, 0.5f };
    static const glm::vec4 defaultBackupVec4 = { 0.5f, 0.5f, 0.5f, 1.f };

    bool ColorEdit4(glm::vec4& col);
}
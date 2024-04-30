#pragma once

#include <glm/glm.hpp>
#include "cuda_includes.hpp"
#include "imgui_gradient/src/Interpolation.hpp"

namespace ImGG
{

struct RawMark
{
    float pos;
    glm::vec4 color;

    RawMark(float pos, glm::vec4 color)
        : pos(pos), color(color)
    {}
};

__host__ __device__ static glm::vec4 interpolateColors(const glm::vec4& col1, const glm::vec4& col2, float t, Interpolation interpolationMode)
{
    switch (interpolationMode)
    {
    case Interpolation::Linear:
        return glm::mix(col1, col2, t);
    default:
        printf("interpolateColors() broke\n");
        return glm::vec4(0, 0, 0, 1);
    }
}

__host__ __device__ static glm::vec4 rampInterpolate(const RawMark* lower, const RawMark* upper, float pos, Interpolation interpolationMode)
{
    if (interpolationMode == Interpolation::Constant)
    {
        return lower->color;
    }

    float t = (pos - lower->pos) / (upper->pos - lower->pos);
    return interpolateColors(lower->color, upper->color, t, interpolationMode);
}

} // namespace ImGG

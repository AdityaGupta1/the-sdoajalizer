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

__host__ __device__ static glm::vec4 rampInterpolate(const RawMark* lower, const RawMark* upper, float pos, Interpolation interpolationMode)
{
    switch (interpolationMode)
    {
    case Interpolation::Linear:
    {
        float t = (pos - lower->pos) / (upper->pos - lower->pos);
        return glm::mix(lower->color, upper->color, t);
    }
    case Interpolation::Constant:
        return lower->color;
    }
}

} // namespace ImGG

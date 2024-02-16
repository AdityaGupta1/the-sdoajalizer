#pragma once

#include <glm/glm.hpp>
#include "cuda_includes.hpp"

namespace ColorUtils
{
	// TODO: use more accurate approximation?
	__host__ __device__ glm::vec4 linearToSrgb(glm::vec4 linearCol)
	{
		return glm::vec4(glm::pow(glm::vec3(linearCol), glm::vec3(0.454545454545454545454545f)), linearCol.a);
	}

	__host__ __device__ glm::vec4 srgbToLinear(glm::vec4 srgbCol)
	{
		return glm::vec4(glm::pow(glm::vec3(srgbCol), glm::vec3(2.2f)), srgbCol.a);
	}
}

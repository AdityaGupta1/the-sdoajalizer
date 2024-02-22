#pragma once

#include <glm/glm.hpp>
#include "cuda_includes.hpp"

namespace ColorUtils
{
	// ==================================================================
	// MISC
	// ==================================================================

	__host__ __device__ inline float luminance(glm::vec3 v)
	{
		return glm::dot(v, glm::vec3(0.2126f, 0.7152f, 0.0722f));
	}

	// ==================================================================
	// SRGB/LINEAR CONVERSION
	// ==================================================================

	// TODO: use more accurate approximation?
	__host__ __device__ inline glm::vec4 linearToSrgb(glm::vec4 linearCol)
	{
		return glm::vec4(glm::pow(glm::vec3(linearCol), glm::vec3(0.454545454545454545454545f)), linearCol.a);
	}

	__host__ __device__ inline glm::vec4 srgbToLinear(glm::vec4 srgbCol)
	{
		return glm::vec4(glm::pow(glm::vec3(srgbCol), glm::vec3(2.2f)), srgbCol.a);
	}

	// ==================================================================
	// TONE MAPPING
	// ==================================================================

	__host__ __device__ inline glm::vec4 reinhard(glm::vec4 col)
	{
		glm::vec3 rgb = glm::vec3(col);
		return glm::vec4(rgb / (1.f + luminance(rgb)), col.a);
	}
}

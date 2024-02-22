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
	__host__ __device__ inline glm::vec3 linearToSrgb(glm::vec3 linearCol)
	{
		return glm::pow(linearCol, glm::vec3(0.454545454545454545454545f));
	}

	__host__ __device__ inline glm::vec3 srgbToLinear(glm::vec3 srgbCol)
	{
		return glm::pow(srgbCol, glm::vec3(2.2f));
	}

	__host__ __device__ inline glm::vec4 srgbToLinear(glm::vec4 srgbCol)
	{
		return glm::vec4(srgbToLinear(glm::vec3(srgbCol)), srgbCol.a);
	}

	// ==================================================================
	// TONE MAPPING
	// ==================================================================

	__host__ __device__ inline glm::vec3 reinhard(glm::vec3 col)
	{
		return col / (1.f + luminance(col));
	}

	// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
	__host__ __device__ inline glm::vec3 ACESFilm(glm::vec3 x)
	{
		float a = 2.51f;
		float b = 0.03f;
		float c = 2.43f;
		float d = 0.59f;
		float e = 0.14f;
		return glm::clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.f, 1.f);
	}
}

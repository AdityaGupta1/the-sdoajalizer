// MIT License
//
// Copyright (c) 2024 Missing Deadlines (Benjamin Wrensch)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp>

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

	__host__ __device__ inline float luminance(glm::vec4 v)
	{
		return luminance(glm::vec3(v));
	}

	// https://mattlockyer.github.io/iat455/documents/rgb-hsv.pdf
	__host__ __device__ inline glm::vec3 rgbToHsv(glm::vec3 v)
	{
		float m_max = glm::compMax(v);
		float m_min = glm::compMin(v);
		float delta = m_max - m_min;

		float H;
		if (delta == 0.f)
		{
			H = 0.f;
		}
		else
		{
			if (m_max == v.r)
			{
				H = fmodf((v.g - v.b) / delta, 6.f);
			}
			else if (m_max == v.g)
			{
				H = (v.b - v.r) / delta + 2.f;
			}
			else
			{
				H = (v.r - v.g) / delta + 4.f;
			}

			H /= 6.f;
		}

		float V = m_max;

		float S;
		if (V == 0.f)
		{
			S = 0.f;
		}
		else
		{
			S = delta / V;
		}

		return glm::vec3(H, S, V);
	}

	// ==================================================================
	// SRGB/LINEAR CONVERSION
	// ==================================================================

	// TODO: use more accurate approximation?
	__host__ __device__ inline glm::vec3 linearToSrgb(glm::vec3 linearCol)
	{
		return glm::pow(linearCol, glm::vec3(0.454545454545454545454545f));
	}

	__host__ __device__ inline glm::vec4 linearToSrgb(glm::vec4 linearCol)
	{
		return glm::vec4(linearToSrgb(glm::vec3(linearCol)), linearCol.a);
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

	// https://iolite-engine.com/blog_posts/minimal_agx_implementation

	// 0: Default, 1: Golden, 2: Punchy
	#define AGX_LOOK 0

	__host__ __device__ inline glm::vec3 _agxDefaultContrastApprox(glm::vec3 x)
	{
		glm::vec3 x2 = x * x;
		glm::vec3 x4 = x2 * x2;

		return +15.5f * x4 * x2
			- 40.14f * x4 * x
			+ 31.96f * x4
			- 6.868f * x2 * x
			+ 0.4298f * x2
			+ 0.1191f * x
			- 0.00232f;
	}

	__host__ __device__ inline glm::vec3 _agx_internal(glm::vec3 val)
	{
		const glm::mat3 agx_mat = glm::mat3(
			0.842479062253094, 0.0423282422610123, 0.0423756549057051,
			0.0784335999999992, 0.878468636469772, 0.0784336,
			0.0792237451477643, 0.0791661274605434, 0.879142973793104
		);

		constexpr float min_ev = -12.47393f;
		constexpr float max_ev = 4.026069f;

		// Input transform (inset)
		val = agx_mat * val;

		// Log2 space encoding
		//val = glm::clamp(glm::log2(val), min_ev, max_ev);
		val = glm::clamp(glm::log(val) * 1.4426950408889634073599246810019f, min_ev, max_ev); // glm::log2() causes indexing errors for some reason
		val = (val - min_ev) / (max_ev - min_ev);

		// Apply sigmoid function approximation
		val = _agxDefaultContrastApprox(val);

		return val;
	}

	__host__ __device__ inline glm::vec3 _agxEotf(glm::vec3 val)
	{
		const glm::mat3 agx_mat_inv = glm::mat3(
			1.19687900512017, -0.0528968517574562, -0.0529716355144438,
			-0.0980208811401368, 1.15190312990417, -0.0980434501171241,
			-0.0990297440797205, -0.0989611768448433, 1.15107367264116
		);

		// Inverse input transform (outset)
		val = agx_mat_inv * val;

		// sRGB IEC 61966-2-1 2.2 Exponent Reference EOTF Display
		// NOTE: We're linearizing the output here. Comment/adjust when
		// *not* using a sRGB render target
		val = srgbToLinear(val);

		return val;
	}

	__host__ __device__ inline glm::vec3 _agxLook(glm::vec3 val, int agxLook)
	{
		float luma = luminance(val);

		// Default
		glm::vec3 offset = glm::vec3(0.0);
		glm::vec3 slope = glm::vec3(1.0);
		glm::vec3 power = glm::vec3(1.0);
		float sat = 1.0;

		if (agxLook == 1)
		{
			// Golden
			slope = glm::vec3(1.0, 0.9, 0.5);
			power = glm::vec3(0.8);
			sat = 0.8;
		}
		else if (agxLook == 2)
		{
			// Punchy
			slope = glm::vec3(1.0);
			power = glm::vec3(1.35, 1.35, 1.35);
			sat = 1.4;
		}

		// ASC CDL
		val = glm::pow(val * slope + offset, power);
		return luma + sat * (val - luma);
	}

	__host__ __device__ inline glm::vec3 AgX(glm::vec3 col, int agxLook)
	{
		col = _agx_internal(col);

		if (agxLook != 0)
		{
			col = _agxLook(col, agxLook);
		}

		col = _agxEotf(col);
		return col;
	}
}

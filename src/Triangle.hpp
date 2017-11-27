#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#ifdef __CUDACC__
#define CUDA_FUNCTION __host__ __device__
#else
#define CUDA_FUNCTION
#endif

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/transform.hpp>

#include "Utils.hpp"

struct Vertex
{
  glm::fvec3 p;
  glm::fvec3 n;
  glm::fvec2 t;

  CUDA_FUNCTION Vertex() : p(0.0f), n(0.0f), t(0.0f) {};
  CUDA_FUNCTION Vertex(const glm::fvec3& pp, const glm::fvec3& nn, const glm::fvec2& tt) : p(pp), n(nn), t(tt) {};
};

struct Triangle {
  Vertex vertices[3];

  CUDA_FUNCTION Triangle(const Vertex v0, const Vertex v1, const Vertex v2) {
		vertices[0] = v0;
		vertices[1] = v1;
		vertices[2] = v2;
	}

  CUDA_FUNCTION Triangle& operator=(const Triangle& that) = default;

  CUDA_FUNCTION inline glm::vec3 min() const {
		return glm::min(glm::min(vertices[0].p, vertices[1].p), vertices[2].p);
	}

  CUDA_FUNCTION inline glm::vec3 max() const {
		return glm::max(glm::max(vertices[0].p, vertices[1].p), vertices[2].p);
	}

  CUDA_FUNCTION glm::vec3 normal() const {
		return glm::normalize(glm::cross(vertices[1].p - vertices[0].p, vertices[2].p - vertices[0].p));
	}
};

#endif

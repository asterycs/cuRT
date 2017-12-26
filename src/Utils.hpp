#ifndef UTILS_HPP
#define UTILS_HPP

#ifdef __CUDACC__
  #define CUDA_FUNCTION __host__ __device__
#else
  #define CUDA_FUNCTION
#endif

#include <string>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <numeric>

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/glm.hpp"

#define WWIDTH 600
#define WHEIGHT 600

class Triangle;
class Model;

#ifndef NDEBUG
  #define GL_CHECK(call) do { \
          call; \
          CheckOpenGLError(#call, __FILE__, __LINE__); \
      } while (0)
#else
  #define GL_CHECK(call) call
#endif

#ifdef ENABLE_CUDA
  #ifndef NDEBUG
    #define CUDA_CHECK(call) do { \
            call;\
            CheckCudaError(#call, __FILE__, __LINE__); \
        } while (0)
  #else
    #define CUDA_CHECK(call) call
  #endif
#else
  #define CUDA_CHECK(call)
#endif

void CheckCudaError(const char* stmt, const char* fname, int line);

void CheckOpenGLError(const char* call, const char* fname, int line);

std::string readFile(const std::string& filePath);

bool fileExists(const std::string& fileName);

struct Material
{
  glm::fvec3 colorAmbient;
  glm::fvec3 colorDiffuse;
  glm::fvec3 colorEmission;
  glm::fvec3 colorSpecular;
  glm::fvec3 colorTransparent;

  float refrIdx;
  float shininess;

  // TextureIndex?

  Material() : colorAmbient(0.0f), colorDiffuse(0.0f), colorEmission(0.0f), colorSpecular(0.0f), shininess(0.0f) {};

};


struct AABB
{
  glm::fvec3 max;
  glm::fvec3 min;

  CUDA_FUNCTION AABB(const glm::fvec3& a, const glm::fvec3& b) : max(glm::max(a, b)), min(glm::min(a, b)) { }
  CUDA_FUNCTION AABB() : max(0.f), min(0.f) { }
  
  CUDA_FUNCTION float area() const;
  CUDA_FUNCTION unsigned int maxAxis() const;
};

struct Node
{
  AABB bbox;
  int startTri;
  int nTri; // exclusive
  int rightIndex;
};

std::unique_ptr<Node> createBVH(const Model& model);

struct MeshDescriptor
{
  std::vector<unsigned int> vertexIds;
  int materialIdx;

  MeshDescriptor(const std::vector<unsigned int> vertexIds, const unsigned int materialId)
  :
    vertexIds(vertexIds),
    materialIdx(materialId) {};

  MeshDescriptor() : vertexIds(), materialIdx(-1) {};
};


struct Ray
{
  glm::fvec3 origin;
  glm::fvec3 direction;
  glm::fvec3 inverseDirection;

  CUDA_FUNCTION Ray(const glm::fvec3& o, const glm::fvec3& d) : origin(o), direction(d), inverseDirection(glm::fvec3(1.f) / d) {};
  CUDA_FUNCTION Ray() = default;
};

struct RaycastResult {
  int triangleIdx;
  float t;
  glm::fvec2 uv;
  glm::fvec3 point;


  CUDA_FUNCTION RaycastResult(const unsigned int i,
    const float t,
    const glm::fvec2& uv,
    const glm::fvec3& point)
    :
    triangleIdx(i),
    t(t),
    uv(uv),
    point(point)
  {}

  CUDA_FUNCTION RaycastResult()
    :
    triangleIdx(-1),
    t(999999.f),
    uv(),
    point()
  {}

  CUDA_FUNCTION operator bool() const { return (triangleIdx != -1); }
};

#endif

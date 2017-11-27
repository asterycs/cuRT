#ifndef UTILS_HPP
#define UTILS_HPP

#ifdef __CUDACC__
  #define CUDA_FUNCTION __host__ __device__
#else
  #define CUDA_FUNCTION
#endif

#include <string>
#include <iostream>

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/glm.hpp"

#define WWIDTH 600
#define WHEIGHT 600
#define WINDOW_MAXWIDTH 1920
#define WINDOW_MAXHEIGHT 1080

class Triangle;

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


struct Vertex
{
  glm::fvec3 p;
  glm::fvec3 n;
  glm::fvec2 t;

  Vertex(void) : p(0.0f), n(0.0f), t(0.0f) {};
  Vertex(const glm::fvec3& pp, const glm::fvec3& nn, const glm::fvec2& tt) : p(pp), n(nn), t(tt) {};
};

struct Material
{
  glm::fvec3 colorAmbient;
  glm::fvec3 colorDiffuse;
  glm::fvec3 colorEmission;
  glm::fvec3 colorSpecular;
  glm::fvec3 colorShininess;
  glm::fvec3 colorTransparent;

  // TextureIndex?

  Material() : colorAmbient(0.0f), colorDiffuse(0.0f), colorEmission(0.0f), colorSpecular(0.0f), colorShininess(0.0f) {};
};

struct MeshDescriptor
{
  unsigned int start;
  unsigned int nTriangles;
  Material material;

  MeshDescriptor(unsigned int start,
      unsigned int nTriangles,
      Material material)
  :
    start(start),
    nTriangles(nTriangles),
    material(material) {};

  MeshDescriptor() : start(0), nTriangles(0), material() {};
};


struct Ray
{
  glm::fvec3 origin;
  glm::fvec3 direction;

  CUDA_FUNCTION Ray(const glm::fvec3 o, const glm::fvec3 d) : origin(o), direction(d) {};
};

struct RaycastResult {
  const int triangleIdx;
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

  CUDA_FUNCTION operator bool() { return triangleIdx != -1; }
};

#endif

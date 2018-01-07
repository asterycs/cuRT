#ifndef CUDARENDERER_HPP
#define CUDARENDERER_HPP

#include <thrust/device_vector.h>

#include "GLDrawable.hpp"
#include "GLLight.hpp"
#include "GLModel.hpp"
#include "GLCanvas.hpp"
#include "Camera.hpp"

#define QUASIRANDOM

#ifdef QUASIRANDOM
#define CURAND_TYPE curandStateScrambledSobol64
#else
#define CURAND_TYPE curandState_t
#endif

class CudaRenderer
{
public:
  CudaRenderer();
  ~CudaRenderer();

  std::vector<glm::fvec3> debugRayTrace(const glm::ivec2 pixelPos, const glm::ivec2 size, const Camera& camera, GLModel& model, GLLight& light);
  std::vector<glm::fvec3> debugPathTrace(const glm::ivec2 pixelPos, const glm::ivec2 size, const Camera& camera, GLModel& model, GLLight& light);

  void rayTraceToCanvas(GLCanvas& canvas, const Camera& camera, GLModel& model, GLLight& light);
  void pathTraceToCanvas(GLCanvas& canvas, const Camera& camera, GLModel& model, GLLight& light);
  void resize(const glm::ivec2& size);
  void reset();

private:
  thrust::device_vector<CURAND_TYPE> curandStateDevVecX; // For area light sampling
  thrust::device_vector<CURAND_TYPE> curandStateDevVecY;

  Camera lastCamera;
  glm::ivec2 lastSize;
  unsigned int currentPath;
};

#endif // CUDARENDERER_HPP

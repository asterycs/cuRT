#ifndef CUDARENDERER_HPP
#define CUDARENDERER_HPP

#include <thrust/device_vector.h>

#include "GLDrawable.hpp"
#include "GLLight.hpp"
#include "GLModel.hpp"
#include "GLCanvas.hpp"
#include "Camera.hpp"

class CudaRenderer
{
public:
  CudaRenderer();
  ~CudaRenderer();

  std::vector<glm::fvec3> debugRay(const glm::ivec2 pixelPos, const glm::ivec2 size, const Camera& camera, GLModel& model, GLLight& light);
  void renderToCanvas(GLCanvas& canvas, const Camera& camera, GLModel& model, GLLight& light);
  void resize(const glm::ivec2& size);

private:
  thrust::device_vector<curandStateScrambledSobol64> curandStateDevVecX; // For area light sampling
  thrust::device_vector<curandStateScrambledSobol64> curandStateDevVecY;
};

#endif // CUDARENDERER_HPP

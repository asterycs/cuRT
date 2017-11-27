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

  void renderToCanvas(GLCanvas& canvas, const Camera& camera, GLModel& model, GLLight& light);

  void mapModel();

  void unmapModel();

  void unregisterModel();

private:
  thrust::device_vector<curandState_t> curandStateDevVecX;
  thrust::device_vector<curandState_t> curandStateDevVecY;
};

#endif // CUDARENDERER_HPP

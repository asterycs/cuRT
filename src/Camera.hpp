#ifndef CAMERA_HPP
#define CAMERA_HPP

#ifdef __CUDACC__
  #define CUDA_HOST_DEVICE __host__ __device__
  #define CUDA_HOST        __host__
#else
  #define CUDA_HOST_DEVICE
  #define CUDA_HOST
#endif

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>

#include "Utils.hpp"


class Camera
{
public:
  CUDA_HOST_DEVICE Camera();
  CUDA_HOST_DEVICE void rotate(const glm::fvec2& r, const float dTime);
  CUDA_HOST_DEVICE void translate(const glm::fvec3& t, const float dt);
  CUDA_HOST_DEVICE void increaseFOV();
  CUDA_HOST_DEVICE void decreaseFOV();
  
  CUDA_HOST_DEVICE glm::fvec2 normalizedImageCoordinateFromPixelCoordinate(const unsigned int x, const unsigned int y, const glm::ivec2 size) const;
  CUDA_HOST_DEVICE glm::fvec3 worldPositionFromNormalizedImageCoordinate(const glm::fvec2& point, const float aspectRatio) const;

  CUDA_HOST_DEVICE const glm::fmat4 getMVP(const glm::ivec2& size) const;
  CUDA_HOST_DEVICE const glm::fmat4 getP(const glm::ivec2& size) const;
  CUDA_HOST_DEVICE const glm::fmat4 getView() const;
  CUDA_HOST_DEVICE const glm::fvec3 getLeft() const;
  CUDA_HOST_DEVICE const glm::fvec3 getUp() const;
  CUDA_HOST_DEVICE const glm::fvec3 getForward() const;
  CUDA_HOST_DEVICE const glm::fvec3& getPosition() const;
  CUDA_HOST_DEVICE       float     getHAngle() const;
  CUDA_HOST_DEVICE       float     getVAngle() const;
  CUDA_HOST_DEVICE       float     getFov() const;

  CUDA_HOST_DEVICE Ray generateRay(const glm::fvec2& point, const float aspectRatio) const;

  friend CUDA_HOST std::ostream& operator<<(std::ostream& os, const Camera& camera);
  friend CUDA_HOST std::istream& operator>>(std::istream& is, Camera& camera);


private:
  float fov;
  float near;
  float far;
  float moveSpeed;
  float mouseSpeed;
  
  glm::fvec3 position;
  float hAngle;
  float vAngle;
};

#endif

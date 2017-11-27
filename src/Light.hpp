#ifndef LIGHT_HPP
#define LIGHT_HPP

#ifdef __CUDACC__
  #define CUDA_HOST_DEVICE __host__ __device__
  #define CUDA_DEVICE               __device__
  #define CUDA_HOST        __host__
#else
  #define CUDA_HOST_DEVICE
  #define CUDA_DEVICE
  #define CUDA_HOST
#endif

#ifdef ENABLE_CUDA
  #include <curand.h>
  #include <curand_kernel.h>
#endif

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>

#include "Utils.hpp"

class Light
{
public:
  CUDA_HOST_DEVICE Light(const glm::mat4& view);
  CUDA_HOST_DEVICE Light(const glm::mat4& view, const glm::vec2& size, const glm::vec3& emission);
  CUDA_HOST_DEVICE Light();
  CUDA_HOST_DEVICE ~Light();
  
  CUDA_HOST_DEVICE const glm::mat4& getModelMat() const;
  CUDA_HOST_DEVICE const glm::fvec2& getSize() const;
  CUDA_HOST_DEVICE const glm::vec3 getPosition() const;
  CUDA_HOST_DEVICE const glm::vec3 getNormal() const;
  CUDA_HOST_DEVICE const glm::vec3& getEmission() const;
  CUDA_HOST_DEVICE bool isEnabled() const;
  CUDA_HOST_DEVICE void enable();
  CUDA_HOST_DEVICE void disable();

  friend CUDA_HOST std::ostream& operator<<(std::ostream& os, const Light& light);
  friend CUDA_HOST std::istream& operator>>(std::istream& is, Light& light);


  template<typename curandState>
  CUDA_DEVICE void sample(float& pdf, glm::vec3& point, curandState& randomState1, curandState& randomState2) const;
private:
  glm::mat4 modelMat;
  glm::fvec2 size;
  glm::vec3 emission;

  bool enabled;
};


template<typename curandState>
CUDA_DEVICE void Light::sample(float& pdf, glm::vec3& point, curandState& randomState1, curandState& randomState2) const
{
  curandState localState1 = randomState1;
  curandState localState2 = randomState2;

  float x = curand_uniform(&localState1);
  float y = curand_uniform(&localState2);

  randomState1 = localState1;
  randomState2 = localState2;

  glm::fvec2 span = glm::fvec2(x * 2.f, y * 2.f);
  glm::fvec2 rf(span.x - 1.f, span.y - 1.f);

  pdf = 1.0f / (size.x * size.y);

  glm::fvec2 rndClip(rf.x * size.x * 0.5f, rf.y * size.y * 0.5f);
  glm::fvec4 p4 = modelMat * glm::vec4(rndClip, 0, 1);
  point = glm::fvec3(p4);
}

#endif // LIGHT_HPP

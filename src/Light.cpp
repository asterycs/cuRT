#include "Light.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_access.hpp>

CUDA_HOST_DEVICE Light::Light(const glm::mat4& mat) : modelMat(mat), size(0.15f, 0.15f), emission(600.f, 600.f, 600.f), enabled(true)
{

}

CUDA_HOST_DEVICE Light::Light(const glm::mat4& mat, const glm::vec2& size, const glm::vec3& emission) : modelMat(mat), size(size), emission(emission), enabled(true)
{

}

CUDA_HOST_DEVICE Light::Light() : modelMat(1.0f), size(0.f, 0.f), emission(0.f, 0.f, 0.f), enabled(false)
{

}

CUDA_HOST_DEVICE Light::~Light()
{
}

CUDA_HOST_DEVICE const glm::mat4& Light::getModelMat() const
{
  return modelMat;
}

CUDA_HOST_DEVICE const glm::fvec2& Light::getSize() const
{
  return size;
}

CUDA_HOST_DEVICE const glm::fvec3 Light::getNormal() const
{
  const glm::fvec3 ret = -glm::normalize(glm::fvec3(modelMat[2]));

  return ret;
}

CUDA_HOST_DEVICE const glm::fvec3 Light::getPosition() const
{
  return glm::fvec3(modelMat[3]);
}

CUDA_HOST_DEVICE const glm::fvec3& Light::getEmission() const
{
  return emission;
}

CUDA_HOST_DEVICE bool Light::isEnabled() const
{
  return enabled;
}

CUDA_HOST_DEVICE void Light::enable()
{
  enabled = true;
}

CUDA_HOST_DEVICE void Light::disable()
{
  enabled = false;
}

CUDA_HOST std::ostream& operator<<(std::ostream& os, const Light& light)
{
  os << light.enabled << " ";
  os << light.emission.x << " " << light.emission.y << " " << light.emission.z << " ";
  os << light.size.x << " " << light.size.y << " ";

  for (int c = 0; c < 4; ++c)
  {
    const glm::vec4 col = light.modelMat[c];

    os << col.x << " " << col.y << " " << col.z << " " << col.w;

    if (c < 3)
      os << " ";
  }

  return os;
}

CUDA_HOST std::istream& operator>>(std::istream& is, Light& light)
{
  is >> light.enabled;
  is >> light.emission.x >> light.emission.y >> light.emission.z;
  is >> light.size.x >> light.size.y;

  for (int c = 0; c < 4; ++c)
  {
    glm::vec4 col;

    is >> col.x >> col.y >> col.z >> col.w;

    light.modelMat[c] = col;
  }

  return is;
}

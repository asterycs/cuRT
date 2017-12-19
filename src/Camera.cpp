#include "Camera.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>

CUDA_HOST_DEVICE Camera::Camera() :
  fov(glm::quarter_pi<float>()),
  near(0.001f),
  far(10.f),
  moveSpeed(0.5f),
  mouseSpeed(0.0001f),
  position(0.f, 0.f, 0.f),
  hAngle(0.0f),
  vAngle(0.0f)
{
    
}

CUDA_HOST_DEVICE const glm::fmat4 Camera::getP(const glm::ivec2& size) const
{
  return glm::perspectiveFov(fov, (float) size.x, (float) size.y, near, far);
}

CUDA_HOST_DEVICE const glm::fmat4 Camera::getMVP(const glm::ivec2& size) const
{
  return getP(size) * getView();
}

CUDA_HOST_DEVICE const glm::vec3 Camera::getLeft() const
{
  glm::vec4 left = glm::vec4(-1.f, 0.f, 0.f, 0.f);
  left = glm::rotateY(left, -vAngle);
  
  return glm::vec3(left);
}

CUDA_HOST_DEVICE const glm::fvec3 Camera::getForward() const
{
  glm::fvec4 forward = glm::fvec4(0.f, 0.f, -1.f, 0);

  forward = glm::rotateX(forward, -hAngle);
  forward = glm::rotateY(forward, -vAngle);

  return glm::normalize(glm::fvec3(forward));
}

CUDA_HOST_DEVICE const glm::fmat4 Camera::getView() const
{
  glm::fmat4 v = glm::lookAt(position, position + getForward(), getUp());

  return v;
}

CUDA_HOST_DEVICE const glm::fvec3 Camera::getUp() const
{
  return glm::normalize(glm::cross(getForward(), getLeft()));
}

CUDA_HOST_DEVICE void Camera::rotate(const glm::fvec2& r, const float dTime)
{
  hAngle += r.y * mouseSpeed / dTime;
  vAngle += r.x * mouseSpeed / dTime;
}

CUDA_HOST_DEVICE void Camera::translate(const glm::fvec2& t, const float dTime)
{
  position = position + dTime * moveSpeed * glm::fvec3(t.x) * getLeft() + dTime * moveSpeed * glm::fvec3(t.y) * getForward();
}

CUDA_HOST_DEVICE const glm::fvec3& Camera::getPosition() const
{
  return position;
}

CUDA_HOST_DEVICE void Camera::increaseFOV()
{
  if (fov < glm::half_pi<float>())
    fov += 0.025f;
}

CUDA_HOST_DEVICE void Camera::decreaseFOV()
{
  if (fov > 0.f)
    fov -= 0.025f;
}

CUDA_HOST_DEVICE Ray Camera::generateRay(const glm::fvec2& point, const float aspectRatio) const
{
  glm::fvec3 ip = glm::tan(fov * 0.5f) * point.x * aspectRatio * getLeft() +
    glm::tan(fov * 0.5f) * point.y * getUp() +
    getPosition() + getForward();

  glm::fvec3 d = ip - getPosition();

  return Ray(position, glm::normalize(d));
}

CUDA_HOST_DEVICE glm::fvec3 Camera::worldPositionFromNormalizedImageCoordinate(const glm::fvec2& point, const float aspectRatio) const
{
  glm::fvec3 ip = glm::tan(fov * 0.5f) * point.x * aspectRatio * getLeft() +
    glm::tan(fov * 0.5f) * point.y * getUp() +
    getPosition() + getForward();

  glm::fvec3 d = ip - getPosition();

  return position + d;
}

CUDA_HOST_DEVICE glm::fvec2 Camera::normalizedImageCoordinateFromPixelCoordinate(const glm::ivec2& pixel, const glm::ivec2& size) const
{
  float pixelWidth  = 2.f / (size.x + 1);
  float pixelHeight = 2.f / (size.y + 1);

  return glm::fvec2(-1.f + 0.5f * pixelWidth + pixel.x * pixelWidth, -1.f + 0.5f * pixelHeight + pixel.y * pixelHeight);
}

CUDA_HOST_DEVICE float Camera::getHAngle() const
{
  return hAngle;
}

CUDA_HOST_DEVICE float Camera::getVAngle() const
{
  return vAngle;
}

CUDA_HOST_DEVICE float Camera::getFov() const
{
  return fov;
}

CUDA_HOST std::ostream& operator<<(std::ostream& os, const Camera& camera)
{
  os << camera.fov << " " << camera.near << " " << camera.far << " "\
      << camera.position.x << " " << camera.position.y << " " << camera.position.z << " "\
      << camera.hAngle << " " << camera.vAngle;

  return os;
}

CUDA_HOST std::istream& operator>>(std::istream& is, Camera& camera)
{
  is >> camera.fov >> camera.near >> camera.far \
      >> camera.position.x >> camera.position.y >> camera.position.z \
      >> camera.hAngle >> camera.vAngle;

  return is;
}


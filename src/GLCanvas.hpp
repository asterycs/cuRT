#ifndef GLCANVAS_HPP
#define GLCANVAS_HPP

#include <vector>

#include <GL/glew.h>
#include <GL/gl.h>

#ifdef ENABLE_CUDA
  #include <cuda_gl_interop.h>
#endif

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>

#include "Utils.hpp"

class GLCanvas
{
public:
  GLCanvas();
  GLCanvas(const glm::ivec2& size);
  GLCanvas(GLCanvas const& that) = delete;
  void operator=(GLCanvas& that) = delete;
  ~GLCanvas();

  void resize(const glm::ivec2 newSize);
  GLuint getTextureID() const;
  glm::ivec2 getSize() const;
  GLenum getInternalFormat() const;

#ifdef ENABLE_CUDA
  void cudaUnmap();
  cudaSurfaceObject_t getCudaMappedSurfaceObject();
#endif

private:
  GLuint textureID;
  glm::ivec2 size;

#ifdef ENABLE_CUDA
  cudaGraphicsResource_t cudaCanvasResource;
  cudaArray_t canvasCudaArray;
#endif
};

#endif

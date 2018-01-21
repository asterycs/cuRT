#ifndef GLTEXTURE_HPP
#define GLTEXTURE_HPP

#include <vector>

#include <GL/glew.h>
#include <GL/gl.h>

#ifdef ENABLE_CUDA
  #include <cuda_gl_interop.h>
#endif

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>

#include "Utils.hpp"

class GLTexture
{
public:
  GLTexture();
  GLTexture(const unsigned char* pixels, const glm::ivec2 size, const GLenum internalFormat = GL_RGBA32F);
  GLTexture(const glm::ivec2& size, const GLenum internalFormat = GL_RGBA32F);
  GLTexture(GLTexture const& that) = delete;
  void operator=(GLTexture& that) = delete;
  ~GLTexture();

  void load(const unsigned char* pixels, const glm::ivec2 size, const GLenum internalFormat = GL_RGBA32F);
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

  GLenum internalFormat;

#ifdef ENABLE_CUDA
  cudaGraphicsResource_t cudaCanvasResource;
  cudaArray_t cudaCanvasArray;
#endif
};

#endif

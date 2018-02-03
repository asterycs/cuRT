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
  GLTexture(const unsigned char* pixels, const glm::ivec2 size);
  GLTexture(const glm::ivec2& size);
  GLTexture(GLTexture const& that) = delete;
  void operator=(GLTexture& that) = delete;
  ~GLTexture();

  void load(const unsigned char* pixels, const glm::ivec2 size);
  void resize(const glm::ivec2 newSize);

  //template <typename T>
  //std::vector<T> getHostData();
  GLuint getTextureID() const;
  glm::ivec2 getSize() const;
  GLenum getInternalFormat() const;
  GLenum getFormat() const;
  GLenum getType() const;

#ifdef ENABLE_CUDA
  void cudaUnmap();
  cudaSurfaceObject_t getCudaMappedSurfaceObject();
#endif

private:
  GLuint textureID;
  glm::ivec2 size;

  GLenum internalFormat;
  GLenum format;
  GLenum type;

#ifdef ENABLE_CUDA
  cudaGraphicsResource_t cudaCanvasResource;
  cudaArray_t cudaCanvasArray;
#endif
};

#endif

#ifndef GLTEXTURE_HPP
#define GLTEXTURE_HPP

#include <GL/glew.h>
#include <GL/gl.h>

#ifdef ENABLE_CUDA
  #include <cuda_gl_interop.h>
#endif

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>

class GLTexture
{
public:
  GLTexture(unsigned char* pixels, const unsigned int width, const unsigned int height);
  GLTexture(GLTexture const& that) = delete;
  void operator=(GLTexture& that) = delete;
  ~GLTexture();

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

#endif /* GLTEXTURE_HPP_ */

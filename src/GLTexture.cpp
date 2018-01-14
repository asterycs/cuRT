#include "GLTexture.hpp"
#include "Triangle.hpp"

#include <glm/gtc/matrix_transform.hpp>

#ifdef ENABLE_CUDA
  #include <cuda_runtime.h>
#endif

GLTexture::GLTexture()
#ifdef ENABLE_CUDA
  : canvasCudaArray()
#endif
{

}

GLTexture::GLTexture(const glm::ivec2& newSize, const GLenum internalFormat)
#ifdef ENABLE_CUDA
  : canvasCudaArray()
#endif
{
  this->size = newSize;

  GL_CHECK(glGenTextures(1, &textureID));
  GL_CHECK(glBindTexture(GL_TEXTURE_2D, textureID));
  
  GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
  GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));

  GL_CHECK(glTexImage2D( // Using glTexImage2D instead of glTexSubImage2D for faster resizing
    GL_TEXTURE_2D,
    0,
    internalFormat,
    size.x,
    size.y,
    0,
    GL_RGBA,
    GL_FLOAT,
    NULL
  ));

  this->internalFormat = internalFormat;
  CUDA_CHECK(cudaGraphicsGLRegisterImage(&cudaCanvasResource, textureID, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
}

GLTexture::~GLTexture ()
{
  CUDA_CHECK(cudaGraphicsUnregisterResource(cudaCanvasResource));
  GL_CHECK(glDeleteTextures(1, &textureID));
}

void GLTexture::resize(const glm::ivec2 newSize)
{
  this->size = newSize;

  CUDA_CHECK(cudaGraphicsUnregisterResource(cudaCanvasResource));
  GL_CHECK(glBindTexture(GL_TEXTURE_2D, textureID));
  GL_CHECK(glTexImage2D(
    GL_TEXTURE_2D,
    0,
    internalFormat,
    size.x,
    size.y,
    0,
    GL_RGBA,
    GL_FLOAT,
    NULL
  ));
  GL_CHECK(glBindTexture(GL_TEXTURE_2D, 0));
  CUDA_CHECK(cudaGraphicsGLRegisterImage(&cudaCanvasResource, textureID, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));
}

GLuint GLTexture::getTextureID() const
{
  return textureID;
}

glm::ivec2 GLTexture::getSize() const
{
  return size;
}

void GLTexture::cudaUnmap()
{
  CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaCanvasResource));
}

#ifdef ENABLE_CUDA
cudaSurfaceObject_t GLTexture::getCudaMappedSurfaceObject()
{
  CUDA_CHECK(cudaGraphicsMapResources(1, &cudaCanvasResource));
  CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&canvasCudaArray, cudaCanvasResource, 0, 0));
  cudaResourceDesc canvasCudaArrayResourceDesc = cudaResourceDesc();
  (void) canvasCudaArrayResourceDesc; // So the compiler shuts up
  {
    canvasCudaArrayResourceDesc.resType = cudaResourceTypeArray;
    canvasCudaArrayResourceDesc.res.array.array = canvasCudaArray;
  }

  cudaSurfaceObject_t canvasCudaSurfaceObj = 0;
  CUDA_CHECK(cudaCreateSurfaceObject(&canvasCudaSurfaceObj, &canvasCudaArrayResourceDesc));

  return canvasCudaSurfaceObj;
}
#endif


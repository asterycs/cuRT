#include "GLTexture.hpp"

GLTexture::GLTexture(unsigned char* pixels, const unsigned int width, const unsigned int height)
{
  GLint previousTexture, newTexture;
  GL_CHECK(glGetIntegerv(GL_TEXTURE_BINDING_2D, &previousTexture));
  GL_CHECK(glGenTextures(1, &newTexture));
  GL_CHECK(glBindTexture(GL_TEXTURE_2D, newTexture));
  GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
  GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
  GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels));

  GL_CHECK(glBindTexture(GL_TEXTURE_2D, previousTexture));
}

GLTexture::~GLTexture()
{
#ifdef ENABLE_CUDA
  CUDA_CHECK(cudaGraphicsUnregisterResource(cudaCanvasResource));
#endif
  GL_CHECK(glDeleteTextures(1, &textureID));
}


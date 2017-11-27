#ifndef GLSHADOWMAP_HPP
#define GLSHADOWMAP_HPP

#include <GL/glew.h>
#include <GL/gl.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>

#include "Utils.hpp"


class GLShadowMap
{
public:
  GLShadowMap(const glm::ivec2& size);
  GLShadowMap(GLShadowMap const& that) = delete;
  void operator=(GLShadowMap& that) = delete;
  ~GLShadowMap();
  
  GLuint getFrameBufferID() const;
  GLuint getDepthTextureID() const;
  
  glm::ivec2 getSize() const;
  GLuint getVaoID() const;
private:
  glm::ivec2 size;
  GLuint frameBufferID;
  GLuint depthTextureID;
  GLuint vaoID;
  GLuint vboID;
  
  bool isOperational;

};

#endif // GLSHADOWMAP_HPP

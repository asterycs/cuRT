#include "GLShadowMap.hpp"

#include <vector>

GLShadowMap::GLShadowMap(const glm::ivec2& size) :
  size(size),
  frameBufferID(0),
  depthTextureID(0),
  vaoID(0),
  vboID(0),
  isOperational(false)
{
  GL_CHECK(glGenFramebuffers(1, &frameBufferID));
  GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID));

  GL_CHECK(glGenTextures(1, &depthTextureID));
  GL_CHECK(glBindTexture(GL_TEXTURE_2D, depthTextureID));
  GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, size.x, size.y, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0));
  GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
  GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
  //GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE));
  GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL));
  GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
  GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));


  GL_CHECK(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTextureID, 0));

  glDrawBuffer(GL_NONE);
  glReadBuffer(GL_NONE);

  if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
  {
    std::cerr << "Couldn't initialize shadow map" << std::endl;
    return;
  }
  
  const std::vector<glm::vec3> contours = {
    glm::vec3(-1.f, -1.f, 0.0f),
    glm::vec3(3.f, -1.f, 0.0f),
    glm::vec3(-1.f, 3.f, 0.0f)
  };

  const std::vector<glm::vec2> texcoords = {
    glm::vec2(0.f, 0.f),
    glm::vec2(2.f, 0.f),
    glm::vec2(0.f, 2.f)
  };

  std::vector<Vertex> vertices;

  for (unsigned int i = 0; i < contours.size(); ++i)
  {
    Vertex v;
    v.p = contours[i];
    v.t = texcoords[i];
    vertices.push_back(v);
  }

  GL_CHECK(glGenVertexArrays(1, &vaoID));
  GL_CHECK(glBindVertexArray(vaoID));

  GL_CHECK(glGenBuffers(1, &vboID));
  GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, vboID));
  GL_CHECK(glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW));

  GL_CHECK(glEnableVertexAttribArray(0));
  GL_CHECK(glVertexAttribPointer(
     0,
     3,
     GL_FLOAT,
     GL_FALSE,
     sizeof(Vertex),
     (GLvoid*)offsetof(Vertex, p)
  ));

  GL_CHECK(glEnableVertexAttribArray(2));
  GL_CHECK(glVertexAttribPointer(
     2,
     2,
     GL_FLOAT,
     GL_FALSE,
     sizeof(Vertex),
     (GLvoid*)offsetof(Vertex, t)
  ));

  GL_CHECK(glBindVertexArray(0));
  GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));

  isOperational = true;
  return;
}

GLShadowMap::~GLShadowMap()
{
  GL_CHECK(glDeleteTextures(1, &depthTextureID));
  GL_CHECK(glDeleteFramebuffers(1, &frameBufferID));
  GL_CHECK(glDeleteBuffers(1, &vboID));
  GL_CHECK(glDeleteVertexArrays(1, &vaoID));
}

glm::ivec2 GLShadowMap::getSize() const
{
  return size;
}

GLuint GLShadowMap::getFrameBufferID() const
{
  return frameBufferID;
}

GLuint GLShadowMap::getDepthTextureID() const
{
  return depthTextureID;
}

GLuint GLShadowMap::getVaoID() const
{
  return vaoID;
}

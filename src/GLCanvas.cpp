#include "GLCanvas.hpp"
#include "Triangle.hpp"

#include <glm/gtc/matrix_transform.hpp>

#ifdef ENABLE_CUDA
  #include <cuda_runtime.h>
#endif

#define INTERNALFORMAT GL_RGBA32F


GLCanvas::GLCanvas(const glm::ivec2& newSize)
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
    INTERNALFORMAT,
    size.x,
    size.y,
    0,
    GL_RGBA,
    GL_FLOAT,
    NULL
  ));

  GL_CHECK(glBindTexture(GL_TEXTURE_2D, 0));

    const std::vector<glm::vec3> contours = {
      glm::fvec3(-1.f, -1.f, 0.0f),
      glm::fvec3(3.f, -1.f, 0.0f),
      glm::fvec3(-1.f, 3.f, 0.0f)
    };

    const std::vector<glm::vec2> texcoords = {
      glm::fvec2(0.f, 0.f),
      glm::fvec2(2.f, 0.f),
      glm::fvec2(0.f, 2.f)
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

  CUDA_CHECK(cudaGraphicsGLRegisterImage(&cudaCanvasResource, textureID, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));

  this->nTriangles = GLuint(vertices.size() / 3);
}

GLCanvas::~GLCanvas ()
{
  CUDA_CHECK(cudaGraphicsUnregisterResource(cudaCanvasResource));
  GL_CHECK(glDeleteTextures(1, &textureID));
  GL_CHECK(glDeleteBuffers(1, &vboID));
  GL_CHECK(glDeleteVertexArrays(1, &vaoID));
}

void GLCanvas::resize(const glm::ivec2 newSize)
{
  this->size = newSize;

  CUDA_CHECK(cudaGraphicsUnregisterResource(cudaCanvasResource));
  GL_CHECK(glBindTexture(GL_TEXTURE_2D, textureID));
  GL_CHECK(glTexImage2D(
    GL_TEXTURE_2D,
    0,
    INTERNALFORMAT,
    size.x,
    size.y,
    0,
    GL_RGBA,
    GL_FLOAT,
    NULL
  ));
  GL_CHECK(glBindTexture(GL_TEXTURE_2D, 0));
  CUDA_CHECK(cudaGraphicsGLRegisterImage(&cudaCanvasResource, textureID, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
}

GLuint GLCanvas::getTextureID() const
{
  return textureID;
}

glm::ivec2 GLCanvas::getSize() const
{
  return size;
}

GLuint GLCanvas::getVaoID() const
{
  return vaoID;
}

GLuint GLCanvas::getVboID() const
{
  return vboID;
}

int GLCanvas::getNTriangles() const
{
  return nTriangles;
}

void GLCanvas::cudaUnmap()
{
  CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaCanvasResource));
}

#ifdef ENABLE_CUDA
cudaSurfaceObject_t GLCanvas::getCudaMappedSurfaceObject()
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


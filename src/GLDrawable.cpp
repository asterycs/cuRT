#include "GLDrawable.hpp"

#include <vector>

#ifdef ENABLE_CUDA
  #include <cuda_runtime.h>
#endif

#include <glm/gtx/string_cast.hpp>

GLDrawable::GLDrawable() :
#ifdef ENABLE_CUDA
  cudaGraphicsTriangleResource(),
  cudaMaterialPtr(),
  cudaTriangleMaterialIdsPtr(),
#endif
  vaoID(0),
  vboID(0),
  nTriangles(0),
  textureInternalFormat(GL_RGB8)
{

}

void GLDrawable::finalizeLoad(const std::vector<Triangle>& triangles, const std::vector<MeshDescriptor>& meshDescriptors, const std::vector<Material>& materials,  const std::vector<unsigned int>& triangleMaterialIds)
{
  if (triangles.size() == 0 || meshDescriptors.size() == 0)
  {
    std::cerr << "Model is empty!" << std::endl;
    return;
  }

  nTriangles = GLuint(triangles.size());
  this->meshDescriptors = meshDescriptors;
  this->materials = materials;

  CUDA_CHECK(cudaMalloc((void**) &cudaMaterialPtr, materials.size() * sizeof(Material)));
  CUDA_CHECK(cudaMemcpy(cudaMaterialPtr, materials.data(), materials.size() * sizeof(Material), cudaMemcpyHostToDevice));

  // gcc 5.4 does not seem to realize that triangleMaterialIds is used here
  (void) triangleMaterialIds;
  CUDA_CHECK(cudaMalloc((void**) &cudaTriangleMaterialIdsPtr, triangleMaterialIds.size() * sizeof(unsigned int)));
  CUDA_CHECK(cudaMemcpy(cudaTriangleMaterialIdsPtr, triangleMaterialIds.data(), triangleMaterialIds.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));

  GL_CHECK(glGenVertexArrays(1, &vaoID));
  GL_CHECK(glBindVertexArray(vaoID));

  GL_CHECK(glGenBuffers(1, &vboID));
  GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, vboID));
  GL_CHECK(glBufferData(GL_ARRAY_BUFFER, triangles.size() * 3 * sizeof(Vertex), triangles.data(), GL_STATIC_DRAW));

  GL_CHECK(glEnableVertexAttribArray(0));
  GL_CHECK(glVertexAttribPointer(
     0,
     3,
     GL_FLOAT,
     GL_FALSE,
     sizeof(Vertex),
     (void*)0
  ));

  GL_CHECK(glEnableVertexAttribArray(1));
  GL_CHECK(glVertexAttribPointer(
     1,
     3,
     GL_FLOAT,
     GL_FALSE,
     sizeof(Vertex),
     (GLvoid*)offsetof(Vertex, n)
  ));

  GL_CHECK(glBindVertexArray(0));

#ifdef ENABLE_CUDA
  registerCuda();
#endif
}

#ifdef ENABLE_CUDA
void GLDrawable::registerCuda()
{
  CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaGraphicsTriangleResource, vboID, cudaGraphicsRegisterFlagsReadOnly));
  CUDA_CHECK(cudaGraphicsMapResources(1, &cudaGraphicsTriangleResource));

  size_t numBytes(0);
  Triangle* cudaTrianglePtr;
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&cudaTrianglePtr, &numBytes, cudaGraphicsTriangleResource));

  if (numBytes / sizeof(Triangle) != nTriangles)
    std::cerr << "Triangle data registration failed: memory mapping failure" << std::endl;

  CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaGraphicsTriangleResource));

}
#endif

GLDrawable::~GLDrawable()
{
	clear();
}

Triangle* GLDrawable::getMappedCudaTrianglePtr()
{
  Triangle* cudaTrianglePtr(nullptr);
  size_t numBytes(0);
  (void) numBytes; // To silence compiler
  CUDA_CHECK(cudaGraphicsMapResources(1, &cudaGraphicsTriangleResource));
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&cudaTrianglePtr, &numBytes, cudaGraphicsTriangleResource));

  return cudaTrianglePtr;
}

void GLDrawable::unmapCudaTrianglePtr()
{
  CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaGraphicsTriangleResource));
}

void GLDrawable::clear()
{
  if (nTriangles > 0)
  {
    CUDA_CHECK(cudaGraphicsUnregisterResource(cudaGraphicsTriangleResource));
    CUDA_CHECK(cudaFree(cudaMaterialPtr));
    CUDA_CHECK(cudaFree(cudaTriangleMaterialIdsPtr));
  }

  GL_CHECK(glBindVertexArray(0));
  GL_CHECK(glDeleteBuffers(1, &vboID));
  GL_CHECK(glDeleteVertexArrays(1, &vaoID));

  vaoID = 0;
  vboID = 0;
  nTriangles = 0;
}

GLuint GLDrawable::getVaoID() const
{
  return vaoID;
}

GLuint GLDrawable::getVboID() const
{
  return vboID;
}

GLuint GLDrawable::getNTriangles() const
{
  return nTriangles;
}

GLuint GLDrawable::getNMeshes() const
{
  return meshDescriptors.size();
}

const std::vector<MeshDescriptor>& GLDrawable::getMeshDescriptors() const
{
  return meshDescriptors;
}

#ifdef ENABLE_CUDA
Material* GLDrawable::getCudaMaterialsPtr()
{
  return cudaMaterialPtr;
}

unsigned int* GLDrawable::getCudaTriangleMaterialIdsPtr()
{
  return cudaTriangleMaterialIdsPtr;
}
#endif

const std::vector<Material>& GLDrawable::getMaterials() const
{
  return materials;
}


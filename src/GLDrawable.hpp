#ifndef GLDRAWABLE_HPP
#define GLDRAWABLE_HPP

#include <vector>

#include <GL/glew.h>
#include <GL/gl.h>

#ifdef ENABLE_CUDA
  #include <cuda_gl_interop.h>
#endif

#include "Triangle.hpp"
#include "Utils.hpp"

class GLDrawable
{
public:
  GLDrawable(const GLDrawable& that) = delete;
  GLDrawable& operator=(const GLDrawable& that) = delete;
  ~GLDrawable();

  GLuint getVaoID() const;
  GLuint getVboID() const;
  GLuint getNTriangles() const;
  GLuint getNMeshes() const;

  const Model& getModel() const;

  const std::vector<MeshDescriptor>& getMeshDescriptors() const; // Used when drawing OpenGL
  const std::vector<Material>& getMaterials() const;

  Triangle* cudaGetMappedTrianglePtr();
  void cudaUnmapTrianglePtr();

  Material* cudaGetMappedMaterialsPtr();
  void cudaUnmapMaterialsPtr();

protected:
  GLDrawable();
  void clear();
  void finalizeLoad(const std::vector<Triangle>& triangles, const std::vector<MeshDescriptor>& meshDescriptors, const std::vector<Material>& materials);

private:

#ifdef ENABLE_CUDA
  void registerCuda();
  cudaGraphicsResource_t cudaGraphicsTriangleResource;
  Material* cudaMaterialPtr;
#endif

  GLuint vaoID;
  GLuint vboID;
  GLuint nTriangles;
  
  GLenum textureInternalFormat;

  std::vector<MeshDescriptor> meshDescriptors;
  std::vector<Material> materials;
};

#endif

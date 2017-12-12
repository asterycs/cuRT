#include "GLModel.hpp"

#ifdef ENABLE_CUDA
  #include <cuda_runtime.h>
#endif

#include <vector>

GLModel::GLModel()
{

}

GLModel::~GLModel()
{
#ifdef ENABLE_CUDA
  cudaFree(deviceBVH);
#endif
}

void GLModel::load(const Model& model)
{
  clear();

  fileName = model.getFileName();

  auto meshDescriptors = std::vector<MeshDescriptor>(model.getMeshDescriptors().begin(), model.getMeshDescriptors().end());
  bvhBoxDescriptors = model.getBVHBoxDescriptors();
  auto materials = model.getMaterials();

#ifdef ENABLE_CUDA
  CUDA_CHECK(cudaMalloc((void**) &deviceBVH, model.getBVH().size() * sizeof(Node)));
  CUDA_CHECK(cudaMemcpy(deviceBVH, model.getBVH().data(), model.getBVH().size() * sizeof(Node), cudaMemcpyHostToDevice));
#endif

  const std::vector<Triangle>& triangles = model.getTriangles();
  
  std::cout << "Triangles: " << triangles.size() << std::endl;
  
  finalizeLoad(triangles, meshDescriptors, materials);
}

const std::string& GLModel::getFileName() const
{
  return fileName;
}

#ifdef ENABLE_CUDA
const Node* GLModel::getDeviceBVH() const
{
  return deviceBVH;
}
#endif

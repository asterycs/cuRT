#include "GLModel.hpp"

#ifdef ENABLE_CUDA
  #include <cuda_runtime.h>
#endif

#include <vector>

GLModel::GLModel() : deviceBVH(nullptr)
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

  auto& meshDescriptors = model.getMeshDescriptors();
  bvhBoxDescriptors = model.getBVHBoxDescriptors();
  bvhBoxMaterials = model.getBVHBoxMaterials();

  auto& materials = model.getMaterials();
  auto& triangleMaterialIds = model.getTriangleMaterialIds();

#ifdef ENABLE_CUDA
  CUDA_CHECK(cudaMalloc((void**) &deviceBVH, model.getBVH().size() * sizeof(Node)));
  CUDA_CHECK(cudaMemcpy(deviceBVH, model.getBVH().data(), model.getBVH().size() * sizeof(Node), cudaMemcpyHostToDevice));
#endif

  const std::vector<Triangle>& triangles = model.getTriangles();
  
  std::cout << "Triangles: " << triangles.size() << std::endl;
  
  finalizeLoad(triangles, meshDescriptors, materials, triangleMaterialIds);
}

const std::vector<MeshDescriptor>& GLModel::getBVHBoxDescriptors() const
{
  return bvhBoxDescriptors;
}

const std::vector<Material>& GLModel::getBVHBoxMaterials() const
{
  return bvhBoxMaterials;
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

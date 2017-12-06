#include "GLModel.hpp"

#include <vector>

GLModel::GLModel()
{

}

GLModel::~GLModel()
{

}

void GLModel::load(const Model& model)
{
  clear();

  fileName = model.getFileName();

  auto meshDescriptors = std::vector<MeshDescriptor>(model.getMeshDescriptors().begin(), model.getMeshDescriptors().end());
  bvhBoxDescriptors = std::vector<MeshDescriptor>(model.getBVHBoxDescriptors().begin(), model.getBVHBoxDescriptors().end());

  const std::vector<Triangle>& triangles = model.getTriangles();
  
  std::cout << "Triangles: " << triangles.size() << std::endl;
  
  finalizeLoad(triangles, meshDescriptors);
}

const std::string& GLModel::getFileName() const
{
  return fileName;
}

const std::vector<MeshDescriptor>& GLModel::getBVHBoxDescriptors() const
{
  return this->bvhBoxDescriptors;
}

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

  std::vector<MeshDescriptor> meshDescriptors = std::vector<MeshDescriptor>(model.getMeshDescriptors().begin(), model.getMeshDescriptors().end());

  const std::vector<Triangle>& triangles = model.getTriangles();
  
  std::cout << "Triangles: " << triangles.size() << std::endl;
  
  finalizeLoad(triangles, meshDescriptors);
}

const std::string& GLModel::getFileName() const
{
  return fileName;
}

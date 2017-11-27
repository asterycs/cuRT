#ifndef MODEL_HPP
#define MODEL_HPP

#include <vector>
#include <string>

#include "assimp/scene.h"

#include "Utils.hpp"
#include "Triangle.hpp"

class Model
{
public:
  Model();
  Model(const aiScene *scene, const std::string& fileName);
  const std::vector<Triangle>& getTriangles() const;
  const std::vector<MeshDescriptor>& getMeshDescriptors() const;
  const std::string& getFileName() const;
private:
  std::vector<Triangle> triangles;
  std::vector<MeshDescriptor> meshInfos;
  std::string fileName;

  AABB boundingBox;
};

#endif

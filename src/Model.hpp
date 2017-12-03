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
  const AABB& getBbox() const;
  void createBVH();
private:
  void initialize(const aiScene *scene);
  std::vector<Triangle> triangles;
  std::vector<MeshDescriptor> meshInfos;
  std::string fileName;

  AABB boundingBox;
  std::unique_ptr<Node> bvh;
};

#endif

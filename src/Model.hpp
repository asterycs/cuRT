#ifndef MODEL_HPP
#define MODEL_HPP

#include <vector>
#include <string>

#include "assimp/scene.h"

#include "Utils.hpp"
#include "Triangle.hpp"
#include "BVHBuilder.hpp"

class Model
{
public:
  Model();
  Model(const aiScene *scene, const std::string& fileName);
  const std::vector<Triangle>& getTriangles() const;
  const std::vector<Material>& getMaterials() const;
  const std::vector<unsigned int>& getTriangleMaterialIds() const;
  const std::vector<MeshDescriptor>& getMeshDescriptors() const;

  const std::vector<Material>& getBVHBoxMaterials() const;
  const std::vector<MeshDescriptor>& getBVHBoxDescriptors() const;
  
  const AABB& getBbox() const;
  const std::vector<Node>& getBVH() const;
  const std::string& getFileName() const;
private:
  void initialize(const aiScene *scene);

  std::vector<Triangle> triangles;
  std::vector<MeshDescriptor> meshDescriptors; // For GL drawing

  std::vector<Material> materials;
  std::vector<unsigned int> triangleMaterialIds;

  std::vector<MeshDescriptor> bvhBoxDescriptors; // For bvh visualization
  std::vector<Material> bvhBoxMaterials;
  std::string fileName;

  AABB boundingBox;
  std::vector<Node> bvh;
};

#endif

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
  const std::vector<Material>& getMaterials() const;
  const std::vector<unsigned int>& getTriangleMaterialIds() const;
  const std::vector<MeshDescriptor>& getMeshDescriptors() const;
  const std::vector<MeshDescriptor>& getBVHBoxDescriptors() const;
  const std::vector<Node>& getBVH() const;
  const std::string& getFileName() const;
  const AABB& getBbox() const;
  void createBVH();
  std::vector<unsigned int> getMortonCodes() const;
private:
  void initialize(const aiScene *scene);
  void createBVHColors();

  std::vector<Triangle> triangles;
  std::vector<MeshDescriptor> meshDescriptors; // For GL drawing

  std::vector<Material> materials;
  std::vector<unsigned int> triangleMaterialIds;

  std::vector<MeshDescriptor> bvhBoxDescriptors; // For bvh visualization
  std::string fileName;

  AABB boundingBox;
  std::vector<Node> bvh;
};

#endif

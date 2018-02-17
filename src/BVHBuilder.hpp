#ifndef BVHBUILDER_HPP
#define BVHBUILDER_HPP

#include <vector>
#include <string>

#include "Utils.hpp"
#include "Triangle.hpp"

#define MAX_TRIS_PER_LEAF 8

enum SplitMode
{
  OBJECT_MEDIAN,
  SAH
};

class BVHBuilder
{
public:
  BVHBuilder();
  ~BVHBuilder();
  
  std::vector<Node> getBVH();
  std::vector<Triangle> getTriangles();
  
  void build(const enum SplitMode splitMode, std::vector<Triangle>& triangles);
  
  unsigned int expandBits(unsigned int v);
  AABB computeBB(const Node node);
  std::vector<unsigned int> getMortonCodes(const std::vector<Triangle>& triangles, const AABB& boundingBox);
  void createBVHColors();
  std::vector<std::pair<Triangle, unsigned int>> sortOnMorton(std::vector<Triangle> triangles, const AABB& boundingBox);
  void reorderMeshIndices(const std::vector<std::pair<Triangle, unsigned int>>& trisWithIds);
  bool isBalanced(const Node *node, const Node* root, int* height);
  void sortTrisOnAxis(const Node& node, const unsigned int axis);
  bool splitNode(const Node& node, Node& leftChild, Node& rightChild, const SplitMode splitMode);
  std::vector<std::pair<Triangle, unsigned int>> createBVH(const enum SplitMode splitMode);
  
private:
  std::vector<Node> bvh;
  std::vector<std::pair<Triangle, unsigned int>> trisWithIds;
  
  std::vector<MeshDescriptor> meshDescriptors;
  std::vector<unsigned int> triangleMaterialIds;
  std::vector<MeshDescriptor> bvhBoxDescriptors;
  std::vector<Material> bvhBoxMaterials;
};

#endif // BVHBUILDER_HPP

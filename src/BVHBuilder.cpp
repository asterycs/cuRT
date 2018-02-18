#include "BVHBuilder.hpp"

#include <stack>
#include <parallel/algorithm>

BVHBuilder::BVHBuilder()
{
  
}

BVHBuilder::~BVHBuilder()
{
  
}

// From https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
unsigned int BVHBuilder::expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

AABB BVHBuilder::computeBB(const Node node)
{
  // Construct BB
  glm::fvec3 minXYZ = glm::fvec3(std::numeric_limits<float>::max());
  glm::fvec3 maxXYZ = glm::fvec3(-std::numeric_limits<float>::max());

  if (node.nTri > 0)
  {
    for (int ti = node.startTri; ti < node.startTri + node.nTri; ++ti)
    {
      glm::fvec3 pmin = trisWithIds[ti].first.min();
      glm::fvec3 pmax = trisWithIds[ti].first.max();

      minXYZ = glm::min(pmin, minXYZ);
      maxXYZ = glm::max(pmax, maxXYZ);
    }
  }
  else
  {
    minXYZ = glm::fvec3(0.f);
    maxXYZ = glm::fvec3(0.f);
  }

  AABB ret(minXYZ, maxXYZ);

  return ret;
}

std::vector<unsigned int> BVHBuilder::getMortonCodes(const std::vector<Triangle>& triangles, const AABB& boundingBox)
{
  // Create Morton codes
  const AABB& modelBbox = boundingBox;
  const glm::vec3 diff = modelBbox.max - modelBbox.min;
  const float maxDiff = std::max(std::max(diff.x, diff.y), diff.z);

  std::vector<unsigned int> mortonCodes(triangles.size());

  std::vector<Triangle> normTris = triangles;

  glm::fvec3 toAdd = -modelBbox.min;

  for (std::size_t d = 0; d < 3; ++d)
  {
    if (toAdd[d] < 0.f)
      toAdd[d] = 0.f;
  }

  for (auto& tri : normTris)
  {
    for (auto& v : tri.vertices)
    {
      v.p += toAdd;
    }
  }

  for (auto& tri : normTris)
  {
    for (auto& v : tri.vertices)
    {
      v.p /= maxDiff;
    }
  }

  // Normalization done, now get the the centroids and encode every dimension as a 10 bit integer
  for (std::size_t triIdx = 0; triIdx < triangles.size(); ++triIdx)
  {
    const auto& tri = triangles[triIdx];

    glm::fvec3 centr(0.f);

    for (const auto& v : tri.vertices)
    {
      centr += v.p;
    }

    centr /= 3.f;


    glm::ivec3 zOrder = glm::ivec3(centr.x * 1023, centr.y * 1023, centr.z * 1023); // 10 bits

    unsigned int xExpanded = expandBits(zOrder.x);
    unsigned int yExpanded = expandBits(zOrder.y);
    unsigned int zExpanded = expandBits(zOrder.z);


    mortonCodes[triIdx] = 4 * xExpanded + 2 * yExpanded + zExpanded;
  }

  return mortonCodes;
}

void BVHBuilder::createBVHColors()
{
  // Assign unique colors to primitives in the same leaf
  std::vector<MeshDescriptor> descriptors;
  std::vector<Material> materials;

  for (const auto& node : bvh)
  {
    if (node.rightIndex == -1) // Is leaf
    {
      float r = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
      float g = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
      float b = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);

      Material material;
      material.colorAmbient = glm::fvec3(r, g, b);
      material.colorDiffuse = glm::fvec3(r, g, b);

      MeshDescriptor descr;
      descr.vertexIds = std::vector<unsigned int>(node.nTri * 3);
      std::iota(descr.vertexIds.begin(), descr.vertexIds.end(), node.startTri * 3);
      descr.materialIdx = materials.size();

      materials.push_back(material);
      descriptors.push_back(descr);
    }
  }

  bvhBoxMaterials = materials;
  bvhBoxDescriptors = descriptors;
}

std::vector<std::pair<Triangle, unsigned int>> BVHBuilder::sortOnMorton(std::vector<Triangle> triangles, const AABB& boundingBox)
{
  std::vector<std::pair<Triangle, unsigned int>> triIdx;
  
  unsigned int idx = 0;
  for (auto t : triangles)
  {
    triIdx.push_back(std::make_pair(t, idx++));
  }
  
  const auto& mortonCodes = getMortonCodes(triangles, boundingBox);

  std::sort(triIdx.begin(), triIdx.end(), [&mortonCodes] (auto l, auto r)
  {
    return mortonCodes[l.second] < mortonCodes[r.second];
  });
  
  return triIdx;
}

void BVHBuilder::reorderTrianglesAndMaterialIds()
{
  std::vector<unsigned int> triIdxMap;
  triIdxMap.resize(trisWithIds.size());

  for (std::size_t i = 0; i < trisWithIds.size(); ++i)
  {
    triIdxMap[trisWithIds[i].second] = i;
  }

  std::vector<unsigned int> orderedTriangleMaterialIds(triangleMaterialIds.size());

  for (std::size_t ti = 0; ti < trisWithIds.size(); ++ti)
  {
    orderedTriangleMaterialIds[ti] = triangleMaterialIds[trisWithIds[ti].second];
  }

  
  triangleMaterialIds = orderedTriangleMaterialIds;

  std::vector<unsigned int> vertIdxMap(triIdxMap.size() * 3);

  for (std::size_t ti = 0; ti < triIdxMap.size(); ++ti)
  {
    for (std::size_t vi = 0; vi < 3; ++vi)
    {
      vertIdxMap[vi + ti * 3] = triIdxMap[ti] * 3 + vi;
    }
  }

  for (auto& m : meshDescriptors)
  {
    for (auto& vi : m.vertexIds)
    {
      vi = vertIdxMap[vi];
    }
  }
  
  return;
}

bool BVHBuilder::isBalanced(const Node *node, const Node* root, int* height)
{
  int lh = 0, rh = 0;

  int l = 0, r = 0;  void reorderIndices(const std::vector<unsigned int>& triRIdxMap);

  if(node->rightIndex == -1)
  {
    *height = 1;
     return 1;
  }

  l = isBalanced(node + 1, root, &lh);
  r = isBalanced(root + node->rightIndex, root, &rh);

  *height = (lh > rh ? lh : rh) + 1;

  if((lh - rh >= 2) || (rh - lh >= 2))
    return 0;

  else return l && r;
}

void BVHBuilder::sortTrisOnAxis(const Node& node, const unsigned int axis)
{
  const auto start = trisWithIds.begin() + node.startTri;
  const auto end = start + node.nTri;

  __gnu_parallel::sort(start, end, [axis](const std::pair<Triangle, unsigned int>& l, const std::pair<Triangle, unsigned int>& r)
      {
        return l.first.center()[axis] < r.first.center()[axis];
      });
}

bool BVHBuilder::splitNode(const Node& node, Node& leftChild, Node& rightChild, const SplitMode splitMode)
{
  if (splitMode == SplitMode::OBJECT_MEDIAN)
	{
    if (node.nTri > static_cast<int>(MAX_TRIS_PER_LEAF))
    {
      leftChild.startTri = node.startTri;
      leftChild.nTri = node.nTri / 2;
      leftChild.bbox = computeBB(leftChild);

      rightChild.startTri = leftChild.startTri + node.nTri / 2;
      rightChild.nTri = node.nTri - leftChild.nTri;
      rightChild.bbox = computeBB(rightChild);

      return true;
    }else
      return false;
	}
	else if (splitMode == SplitMode::SAH)
	{
    if (node.nTri <= static_cast<int>(MAX_TRIS_PER_LEAF))
      return false;

		const float sa = node.bbox.area();
		const float parentCost = node.nTri * sa;

		float minCost = std::numeric_limits<float>::max();
		int minStep = -1;

		const unsigned int a = node.bbox.maxAxis();
    sortTrisOnAxis(node, a);

    AABB fBox = computeBB(node);
    std::vector<AABB> fBoxes(node.nTri - 1);

    const int fStart = node.startTri;
    const int fEnd = node.startTri + node.nTri - 1;

    for (int i = fStart; i < fEnd; ++i)
    {
      fBox.add(trisWithIds[i].first);
      fBoxes[i - node.startTri] = fBox;
    }

    AABB rBox = computeBB(node);
    std::vector<AABB> rBoxes(node.nTri - 1);

    for (int i = fEnd - 1; i > fStart - 1; --i)
    {
      rBox.add(trisWithIds[i].first);
      rBoxes[i - node.startTri] = rBox;
    }

#pragma omp parallel for
    for (int s = 1; s < node.nTri - 1; ++s)
    {
      const float currentCost = fBoxes[s - 1].area() * s + rBoxes[s - 1].area() * (node.nTri - s);

#pragma omp critical
      if (currentCost < minCost)
      {
        minCost = currentCost;
        minStep = s;
      }
    }

		if (minCost < parentCost)
		{
			leftChild.startTri = node.startTri;
			leftChild.nTri = minStep;
			leftChild.bbox = fBoxes[minStep - 1];

			rightChild.startTri = node.startTri + minStep;
			rightChild.nTri = node.nTri - minStep;
			rightChild.bbox = rBoxes[minStep - 1];

			return true;
		}else
		{
		  return false;
		}

	}else
	  throw std::runtime_error("Unknown BVH split type");
}

void BVHBuilder::build(const enum SplitMode splitMode, const std::vector<Triangle>& triangles, const std::vector<unsigned int>& triangleMaterialIds, const std::vector<MeshDescriptor>& meshDescriptors)
{
  this->triangleMaterialIds = triangleMaterialIds;
  this->meshDescriptors = meshDescriptors;
  
  unsigned int idx = 0;

  for (auto t : triangles)
  {
    trisWithIds.push_back(std::make_pair(t, idx++));
  }
  
  Node root;
  root.startTri = 0;
  root.nTri = triangles.size();
  root.bbox = computeBB(root);
  root.rightIndex = -1;
  
  
  // This is a simple top down approach that places the nodes in an array.
  // This makes the transfer to GPU simple.
  std::stack<Node> stack;
  std::stack<int> parentIndices;

  std::vector<Node> finishedNodes;
  std::vector<int> touchCount;

  const unsigned int nodecountAppr = 0;
  finishedNodes.reserve(nodecountAppr);
  touchCount.reserve(nodecountAppr);

  int leafCount = 0;
  int nodeCount = 0;

  stack.push(root);
  parentIndices.push(-1);

  while (!stack.empty()) {

    Node node = stack.top();
    stack.pop();
    int parentIndex = parentIndices.top();
    parentIndices.pop();

    Node left, right;
    const bool wasSplit = splitNode(node, left, right, splitMode);

    if (wasSplit)
    {
      stack.push(right);
      stack.push(left);

      parentIndices.push(nodeCount);
      parentIndices.push(nodeCount);
    }
    else
    {
      ++leafCount;
      node.rightIndex = -1;
    }

    finishedNodes.push_back(node);

    touchCount.push_back(0);
    touchCount[nodeCount] = 0;
    ++nodeCount;

    if (parentIndex != -1)
    {
      ++touchCount[parentIndex];

      if (touchCount[parentIndex] == 2)
      {
        finishedNodes[parentIndex].rightIndex = nodeCount - 1;
      }
    }

  }

  this->bvh = finishedNodes;
  
  reorderTrianglesAndMaterialIds();
}

std::vector<Node> BVHBuilder::getBVH()
{
  return this->bvh;
}

std::vector<Triangle> BVHBuilder::getTriangles()
{
  std::vector<Triangle> triangles(trisWithIds.size());
  
  for (unsigned int i = 0; i < trisWithIds.size(); ++i)
    triangles[i] = trisWithIds[i].first;
    
  return triangles;
}

std::vector<unsigned int> BVHBuilder::getTriangleMaterialIds()
{
  return triangleMaterialIds;
}

std::vector<MeshDescriptor> BVHBuilder::getMeshDescriptors()
{
  return meshDescriptors;
}


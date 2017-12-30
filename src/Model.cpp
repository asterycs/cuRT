#include "Model.hpp"

#include <numeric>
#include <memory>
#include <stack>
#include <cmath>

#include <glm/gtx/string_cast.hpp>

#include "Utils.hpp"

glm::fvec3 ai2glm3f(aiColor3D v)
{
  return glm::fvec3(v[0], v[1], v[2]);
}

Model::Model()
{
  
}

Model::Model(const aiScene *scene, const std::string& fileName) : fileName(fileName)
{
  initialize(scene);
  auto trisWithIds = createBVH(SplitMode::SAH);
  reorderMeshIndices(trisWithIds);
  createBVHColors();
}

void Model::initialize(const aiScene *scene)
{
  std::cout << "Creating model with " << scene->mNumMeshes << " meshes" << std::endl;
  
  unsigned int triangleOffset = 0;
  
  glm::fvec3 maxTri(-999.f,-999.f,-999.f);
  glm::fvec3 minTri(999.f,999.f,999.f);

  for (std::size_t mi = 0; mi < scene->mNumMeshes; mi++)
  {
    aiMesh *mesh = scene->mMeshes[mi];

    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    if (mesh->mMaterialIndex > 0)
    {
      Material material = Material();

      aiMaterial& mat = *scene->mMaterials[mesh->mMaterialIndex];

      aiColor3D aiAmbient    (0.f,0.f,0.f);
      aiColor3D aiDiffuse    (0.f,0.f,0.f);
      aiColor3D aiSpecular   (0.f,0.f,0.f);
      aiColor3D aiEmission   (0.f,0.f,0.f);
      aiColor3D aiTransparent(0.f,0.f,0.f);

      mat.Get(AI_MATKEY_COLOR_AMBIENT,     aiAmbient);
      mat.Get(AI_MATKEY_COLOR_DIFFUSE,     aiDiffuse);
      mat.Get(AI_MATKEY_COLOR_SPECULAR,    aiSpecular);
      mat.Get(AI_MATKEY_COLOR_EMISSIVE,    aiEmission);
      mat.Get(AI_MATKEY_COLOR_TRANSPARENT, aiTransparent);

      mat.Get(AI_MATKEY_REFRACTI,          material.refrIdx);
     // mat.Get(AI_MATKEY_SHININESS,         material.shininess);

      material.colorAmbient     = ai2glm3f(aiAmbient);
      material.colorDiffuse     = ai2glm3f(aiDiffuse);
      //material.colorEmission    = ai2glm3f(aiEmission);
      material.colorSpecular    = ai2glm3f(aiSpecular);
      material.colorTransparent = glm::fvec3(1.f) - ai2glm3f(aiTransparent);

      std::vector<unsigned int> vertexIds(mesh->mNumFaces * 3);
      std::iota(vertexIds.begin(), vertexIds.end(), triangleOffset * 3);

      MeshDescriptor meshDescr = MeshDescriptor(vertexIds, materials.size());
      materials.push_back(material);
      meshDescriptors.push_back(meshDescr);
      triangleOffset += mesh->mNumFaces;
    }else
      continue;

    for (std::size_t vi = 0; vi < mesh->mNumVertices; vi++)
    {
      Vertex newVertex;
      auto& oldVertex = mesh->mVertices[vi];

      newVertex.p = glm::fvec3(oldVertex.x, oldVertex.y, oldVertex.z);

      if (mesh->HasNormals())
      {
        auto& oldNormal = mesh->mNormals[vi];
        newVertex.n = glm::fvec3(oldNormal.x, oldNormal.y, oldNormal.z);
      }

      if (mesh->mTextureCoords[0])
      {
        auto& tc = mesh->mTextureCoords[0][vi];
        newVertex.t = glm::fvec2(tc.x, tc.y);
      }

      vertices.push_back(newVertex);
    }

    for (std::size_t i = 0; i < mesh->mNumFaces; ++i)
    {
      aiFace& face = mesh->mFaces[i];

      if (face.mNumIndices == 3)
      {
        Triangle triangle = Triangle(vertices[face.mIndices[0]], vertices[face.mIndices[1]], vertices[face.mIndices[2]]);

        triangles.push_back(triangle);

        maxTri = glm::max(maxTri, triangle.max());
        minTri = glm::min(minTri, triangle.min());
      }

      triangleMaterialIds.push_back(materials.size() - 1);
    }
  }
}

const std::vector<Triangle>& Model::getTriangles() const
{
  return triangles;
}

const std::vector<MeshDescriptor>& Model::getMeshDescriptors() const
{
  return meshDescriptors;
}

const std::vector<Material>& Model::getMaterials() const
{
  return materials;
}

const std::vector<MeshDescriptor>& Model::getBVHBoxDescriptors() const
{
  return bvhBoxDescriptors;
}

const std::vector<unsigned int>& Model::getTriangleMaterialIds() const
{
  return triangleMaterialIds;
}

const std::string& Model::getFileName() const
{
  return fileName;
}

const AABB& Model::getBbox() const
{
  return this->boundingBox;
}

// From https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

AABB computeBB(const unsigned int startIdx, const unsigned int endIdx, const std::vector<std::pair<Triangle, unsigned int>>& triangles)
{
  // Construct BB
  glm::fvec3 minXYZ = glm::fvec3(std::numeric_limits<float>::max());
  glm::fvec3 maxXYZ = glm::fvec3(-std::numeric_limits<float>::max());

  if (endIdx - startIdx > 0)
  {
    for (unsigned int ti = startIdx; ti < endIdx; ++ti)
    {
      glm::fvec3 pmin = triangles[ti].first.min();
      glm::fvec3 pmax = triangles[ti].first.max();

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

std::vector<unsigned int> getMortonCodes(const std::vector<Triangle>& triangles, const AABB& boundingBox)
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

void Model::createBVHColors()
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

std::vector<std::pair<Triangle, unsigned int>> sortOnMorton(std::vector<Triangle> triangles, const AABB& boundingBox)
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

void Model::reorderMeshIndices(const std::vector<std::pair<Triangle, unsigned int>>& trisWithIds)
{
  std::vector<unsigned int> triIdxMap;
  triIdxMap.resize(trisWithIds.size());

  for (std::size_t i = 0; i < trisWithIds.size(); ++i)
  {
    triIdxMap[trisWithIds[i].second] = i;
  }

  std::vector<Triangle> newTriangles(triangles.size());

  for (std::size_t ti = 0; ti < triangles.size(); ++ti)
  {
    newTriangles[ti] = trisWithIds[ti].first;
  }

  std::vector<unsigned int> newTriangleMaterialIds(triangleMaterialIds.size());

  for (std::size_t ti = 0; ti < triangles.size(); ++ti)
  {
    newTriangleMaterialIds[ti] = triangleMaterialIds[trisWithIds[ti].second];
  }

  triangleMaterialIds = newTriangleMaterialIds;
  triangles = newTriangles;

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
}

bool isBalanced(const Node *node, const Node* root, int* height)
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

void sortTrisOnAxis(const Node& node, const unsigned int axis, std::vector<std::pair<Triangle, unsigned int>>& triangles)
{
  const auto start = triangles.begin() + node.startTri;
  const auto end = start + node.nTri;

  std::sort(start, end, [axis](std::pair<Triangle, unsigned int>& l, std::pair<Triangle, unsigned int>& r)
      {
        return l.first.center()[axis] < r.first.center()[axis];
      });
}

bool splitNode(const Node& node, Node& leftChild, Node& rightChild, const SplitMode splitMode, std::vector<std::pair<Triangle, unsigned int>>& triangles, const unsigned int maxTris)
{
  if (splitMode == SplitMode::OBJECT_MEDIAN)
	{
    if (node.nTri > static_cast<int>(maxTris))
    {
      leftChild.startTri = node.startTri;
      leftChild.nTri = node.nTri / 2;
      leftChild.bbox = computeBB(leftChild.startTri, leftChild.startTri + leftChild.nTri, triangles);

      rightChild.startTri = leftChild.startTri + node.nTri / 2;
      rightChild.nTri = node.nTri - leftChild.nTri;
      rightChild.bbox = computeBB(rightChild.startTri, rightChild.startTri + rightChild.nTri, triangles);

      return true;
    }else
      return false;
	}
	else if (splitMode == SplitMode::SAH)
	{
    if (node.nTri <= static_cast<int>(maxTris))
      return false;

		const float sa = node.bbox.area();
		const float parentCost = node.nTri * sa;

		float minCost = std::numeric_limits<float>::max();
		int minStep = -1;
		AABB minLbox;
		AABB minRbox;

		const unsigned int a = node.bbox.maxAxis();
    sortTrisOnAxis(node, a, triangles);

    for (int s = 1; s < node.nTri - 1; ++s)
    {
      const AABB lBox = computeBB(node.startTri, node.startTri + s, triangles);
      const AABB rBox = computeBB(node.startTri + s, node.startTri + node.nTri, triangles);

      const float currentCost = lBox.area() * s + rBox.area() * (node.nTri - s);

      if (currentCost < minCost)
      {
        minCost = currentCost;
        minStep = s;
        minLbox = lBox;
        minRbox = rBox;
      }
    }

		if (minCost < parentCost)
		{
			leftChild.startTri = node.startTri;
			leftChild.nTri = minStep;
			leftChild.bbox = minLbox;

			rightChild.startTri = node.startTri + minStep;
			rightChild.nTri = node.nTri - minStep;
			rightChild.bbox = minRbox;

			return true;
		}else
		{
		  return false;
		}

	}else
	  throw std::runtime_error("Unknown BVH split type");
}

std::vector<std::pair<Triangle, unsigned int>> Model::createBVH(const enum SplitMode splitMode)
{
  std::vector<std::pair<Triangle, unsigned int>> triIdx;
  
  if (splitMode == SplitMode::OBJECT_MEDIAN)
  {
    triIdx = sortOnMorton(triangles, boundingBox);
  }else
  {
    unsigned int idx = 0;

    for (auto t : triangles)
    {
      triIdx.push_back(std::make_pair(t, idx++));
    }
  }
  
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

  Node first;
  first.startTri = 0;
  first.nTri = triangles.size();
  first.bbox = computeBB(first.startTri, first.startTri + first.nTri, triIdx);
  first.rightIndex = -1;

  stack.push(first);
  parentIndices.push(-1);


  while (!stack.empty()) {

    Node node = stack.top();
    stack.pop();
    int parentIndex = parentIndices.top();
    parentIndices.pop();

    Node left, right;
    const bool splitted = splitNode(node, left, right, splitMode, triIdx, MAX_TRIS_PER_LEAF);

    if (splitted)
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

  return triIdx;
}

const std::vector<Node>& Model::getBVH() const
{
  return bvh;
}

const std::vector<Material>& Model::getBVHBoxMaterials() const
{
  return bvhBoxMaterials;
}

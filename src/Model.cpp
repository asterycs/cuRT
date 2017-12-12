#include "Model.hpp"

#include <numeric>
#include <memory>
#include <queue>
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
  //createBVH();
  //createBVHColors();
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
        aiColor3D aiShininess  (0.f,0.f,0.f);
        aiColor3D aiTransparent(0.f,0.f,0.f);

        mat.Get(AI_MATKEY_COLOR_AMBIENT,     aiAmbient);
        mat.Get(AI_MATKEY_COLOR_DIFFUSE,     aiDiffuse);
        mat.Get(AI_MATKEY_COLOR_SPECULAR,    aiSpecular);
        mat.Get(AI_MATKEY_COLOR_EMISSIVE,    aiEmission);
        mat.Get(AI_MATKEY_SHININESS,         aiShininess);
        mat.Get(AI_MATKEY_COLOR_TRANSPARENT, aiTransparent);

        material.colorAmbient    = ai2glm3f(aiAmbient);
        material.colorDiffuse    = ai2glm3f(aiDiffuse);
        material.colorEmission   = ai2glm3f(aiEmission);
        material.colorSpecular   = ai2glm3f(aiSpecular);
        material.colorShininess  = ai2glm3f(aiShininess);
        material.colorTransparent= ai2glm3f(aiTransparent);

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

AABB computeBB(Node& node, const std::vector<Triangle>& triangles)
{
  // Construct BB
  glm::fvec3 minXYZ = glm::fvec3(std::numeric_limits<float>::max());
  glm::fvec3 maxXYZ = glm::fvec3(-std::numeric_limits<float>::max());

  if (node.nTri > 0)
  {
    for (int ti = node.startTri; ti < node.startTri + node.nTri; ti++)
    {
      glm::fvec3 pmin = triangles[ti].min();
      glm::fvec3 pmax = triangles[ti].max();

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

std::vector<unsigned int> Model::getMortonCodes() const
{
  // Create Morton codes
  const auto& triangles = getTriangles();
  const AABB& modelBbox = boundingBox;
  const glm::vec3 diff = modelBbox.max - modelBbox.min;
  const float maxDiff = std::max(std::max(diff.x, diff.y), diff.z);

  std::vector<unsigned int> mortonCodes(triangles.size());

  std::vector<Triangle> normTris;
  std::copy(triangles.begin(), triangles.end(), std::back_inserter(normTris));

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
/*
void Model::createBVHColors()
{
  // Assign unique colors to primitives in the same leaf. Useful when debugging.
  std::vector<MeshDescriptor> descriptors;

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
      descr.start = node.startTri;
      descr.nTriangles = node.nTri;
      descr.material = material;

      descriptors.push_back(descr);
    }
  }

  bvhBoxDescriptors = descriptors;
}
*/
void Model::createBVH()
{
  auto mortonCodes = getMortonCodes();
  std::vector<unsigned int> sortedMortonCodes;
  std::vector<std::size_t> origIndices; // Keep track of original position

  sort(mortonCodes, sortedMortonCodes, origIndices);

  // This is a simple top down approach that places the nodes in an array.
  // This makes the transfer to GPU simple.
  std::queue<Node> queue;
  std::queue<int> parentIndices;

  std::vector<Node> finishedNodes;
  std::vector<int> touchCount;
  const unsigned int nTris = triangles.size();
  const unsigned int nodecountAppr = std::pow(2.f, std::ceil(std::log2(nTris / 8.f)) + 1) - 1;;
  finishedNodes.reserve(nodecountAppr);
  touchCount.reserve(nodecountAppr);

  int leafCount = 0;
  int nodeCount = 0;

  Node first;
  first.startTri = 0;
  first.nTri = triangles.size();
  first.bbox = computeBB(first, triangles);
  first.rightIndex = -1;

  queue.push(first);
  parentIndices.push(-1);


  while (!queue.empty()) {

    Node node = queue.front();
    queue.pop();
    int parentIndex = parentIndices.front();
    parentIndices.pop();

    int tris = node.nTri - node.startTri;

    if (tris > 8)
    {
      Node left, right;
      left.startTri = node.startTri;
      left.nTri = tris / 2;
      left.bbox = computeBB(left, triangles);

      right.startTri = left.startTri + tris / 2;
      right.nTri = node.nTri - right.startTri;
      right.bbox = computeBB(right, triangles);

      queue.push(left);
      queue.push(right);

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

  ////////////////////////////////////
  // Reorder MeshDescriptors.vertexIds
  ////////////////////////////////////

  return;
}

const std::vector<Node>& Model::getBVH() const
{
  return bvh;
}


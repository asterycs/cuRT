#include "Model.hpp"

#include <numeric>
#include <memory>

#include <glm/gtx/string_cast.hpp>

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
  createBVH();
}

void Model::initialize(const aiScene *scene)
{
  std::cout << "Creating model with " << scene->mNumMeshes << " meshes" << std::endl;
  
  unsigned int triangleOffset = 0;
  
  glm::fvec3 maxTri(-999.f,-999.f,-999.f);
  glm::fvec3 minTri(999.f,999.f,999.f);

    for (unsigned int mi = 0; mi < scene->mNumMeshes; mi++)
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

        MeshDescriptor meshInfo = MeshDescriptor(triangleOffset, mesh->mNumFaces, material);
        meshInfos.push_back(meshInfo);
        triangleOffset += mesh->mNumFaces;
      }
      
      for (unsigned int vi = 0; vi < mesh->mNumVertices; vi++)
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

      for (unsigned int i = 0; i < mesh->mNumFaces; ++i)
      {
        aiFace& face = mesh->mFaces[i];

        if (face.mNumIndices == 3)
        {
          Triangle triangle = Triangle(vertices[face.mIndices[0]], vertices[face.mIndices[1]], vertices[face.mIndices[2]]);

          triangles.push_back(triangle);

          maxTri = glm::max(maxTri, triangle.max());
          minTri = glm::min(minTri, triangle.min());
        }
      }
    }

  boundingBox = AABB(maxTri, minTri);
}

const std::vector<Triangle>& Model::getTriangles() const
{
  return triangles;
}

const std::vector<MeshDescriptor>& Model::getMeshDescriptors() const
{
  return meshInfos;
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

AABB computeBB(Node& node, std::vector<Triangle>& triangles, std::vector<unsigned int>& sortedMorton)
{
  // Construct BB
  glm::fvec3 minXYZ = glm::fvec3(std::numeric_limits<float>::max());
  glm::fvec3 maxXYZ = glm::fvec3(-std::numeric_limits<float>::max());

  if (node.endTri - node.startTri > 0)
  {
    for (int ti = node.startTri; ti < node.endTri; ti++)
    {
      unsigned int triIdx = sortedMorton[ti];
      glm::fvec3 pmin = triangles[triIdx].min();
      glm::fvec3 pmax = triangles[triIdx].max();

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

void Model::createBVH()
{
  // This is just a temporary CPU version. I'll move it to the GPU at some point.
  const AABB& modelBbox = boundingBox;
  const glm::vec3 diff = modelBbox.max - modelBbox.min;
  const float maxDiff = std::max(std::max(diff.x, diff.y), diff.z);

  std::vector<unsigned int> mortonCodes(getTriangles().size());
  std::vector<unsigned int> mortonSortedTriIds(mortonCodes.size());
  std::iota(mortonSortedTriIds.begin(), mortonSortedTriIds.end(), 0);

  std::vector<Triangle> normTris;
  std::copy(getTriangles().begin(), getTriangles().end(), std::back_inserter(normTris));

  glm::fvec3 toAdd = -modelBbox.min;

  for (unsigned int d = 0; d < 3; ++d)
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
  for (unsigned int triIdx = 0; triIdx < getTriangles().size(); ++triIdx)
  {
    const auto& tri = getTriangles()[triIdx];

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


  std::sort(mortonSortedTriIds.begin(), mortonSortedTriIds.end(), [&mortonCodes](unsigned int& l, unsigned int& r) -> bool
      {
        return mortonCodes[l] < mortonCodes[r];
      });


  std::array<Node*, 128> stack;
  int stackIdx = 1;

  std::unique_ptr<Node> root = std::make_unique<Node>();
  root->startTri = 0;
  root->endTri = getTriangles().size();
  root->bbox = modelBbox;

  stack[0] = root.get();

  while (stackIdx > 0)
  {
    --stackIdx;

    Node* currentNode = stack[stackIdx];

    const unsigned int nTris = currentNode->endTri - currentNode->startTri;

    if (nTris > 8)
    {
      currentNode->leftChild = std::make_unique<Node>();
      currentNode->rightChild = std::make_unique<Node>();

      currentNode->leftChild->startTri = currentNode->startTri;
      currentNode->leftChild->endTri = currentNode->startTri + nTris / 2;

      currentNode->rightChild->startTri = currentNode->startTri + nTris / 2 + 1;
      currentNode->rightChild->endTri = currentNode->endTri;

      currentNode->leftChild->bbox = computeBB(*currentNode->leftChild, triangles, mortonSortedTriIds);
      currentNode->rightChild->bbox = computeBB(*currentNode->leftChild, triangles, mortonSortedTriIds);

      stack[stackIdx] = &*currentNode->leftChild;
      ++stackIdx;
      stack[stackIdx] = &*currentNode->rightChild;
      ++stackIdx;
    }
  }

  this->bvh = std::move(root);

/*
  glm::vec3 lmax(-9999.f);
  glm::vec3 lmin(9999.f);

  for (auto& tri : normTris)
  {
    lmax = glm::max(tri.max(), lmax);
    lmin = glm::min(tri.min(), lmin);
  }

  std::cout << glm::to_string(lmax) << std::endl;
  std::cout << glm::to_string(lmin) << std::endl;
*/
  return;
}


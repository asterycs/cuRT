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
  
  BVHBuilder bvhbuilder;
  bvhbuilder.build(SplitMode::SAH, triangles);
  
  this->bvh = bvhbuilder.getBVH();
  this->triangles = bvhbuilder.getTriangles();
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
      mat.Get(AI_MATKEY_SHININESS,         material.shininess);

      material.colorAmbient     = ai2glm3f(aiAmbient);
      material.colorDiffuse     = ai2glm3f(aiDiffuse);
      //material.colorEmission    = ai2glm3f(aiEmission);
      material.colorSpecular    = ai2glm3f(aiSpecular);
      material.colorTransparent = glm::sqrt(glm::fvec3(1.f) - ai2glm3f(aiTransparent));

      int sm;
      mat.Get(AI_MATKEY_SHADING_MODEL, sm);

      switch (sm)
      {
      case aiShadingMode_Gouraud:
        material.shadingMode = material.GORAUD;
        break;
      case aiShadingMode_Fresnel:
        material.shadingMode = material.FRESNEL;
        break;
      default:
        material.shadingMode = material.PHONG;
      }


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

const std::vector<Node>& Model::getBVH() const
{
  return bvh;
}

const std::vector<Material>& Model::getBVHBoxMaterials() const
{
  return bvhBoxMaterials;
}

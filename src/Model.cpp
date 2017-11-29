#include "Model.hpp"

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

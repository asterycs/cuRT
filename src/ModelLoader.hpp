#ifndef MODELLOADER_HPP
#define MODELLOADER_HPP

#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"

#include "Model.hpp"

class ModelLoader
{
public:
  ModelLoader();
  ~ModelLoader();
  
  Model loadOBJ(const std::string& path);
  
private:
  Assimp::Importer importer;
};

#endif // SCENELOADER_HPP

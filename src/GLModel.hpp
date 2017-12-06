#ifndef GLMODEL_HPP
#define GLMODEL_HPP

#include "GLDrawable.hpp"
#include "Model.hpp"
#include "Utils.hpp"

class GLModel : public GLDrawable
{
public:
  GLModel();
  GLModel(const GLModel& that) = delete;
  GLModel& operator=(const GLModel& that) = delete;
  ~GLModel();

  const std::vector<MeshDescriptor>& getBVHBoxDescriptors() const;
  void load(const Model& model);
  const std::string& getFileName() const;

private:
  std::vector<MeshDescriptor> bvhBoxDescriptors;
  std::string fileName;
};

#endif

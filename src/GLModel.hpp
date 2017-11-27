#ifndef GLMODEL_HPP
#define GLMODEL_HPP

#include "GLDrawable.hpp"
#include "Model.hpp"

class GLModel : public GLDrawable
{
public:
  GLModel();
  GLModel(const GLModel& that) = delete;
  GLModel& operator=(const GLModel& that) = delete;
  ~GLModel();

  void load(const Model& model);
  const std::string& getFileName() const;

private:
  std::string fileName;
};

#endif

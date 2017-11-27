#include "CudaRenderer.hpp"

#include <GL/glew.h>
#include <GL/gl.h>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include "Utils.hpp"
#include "Triangle.hpp"

#define BLOCKWIDTH 8
#define EPSILON 0.000001f
#define BIGT 99999.f
#define SHADOWSAMPLING 64
#define RECURSIONS 2

__device__ glm::fvec3 mirrorDirection(const glm::vec3& normal, const glm::vec3& incoming) {
  glm::vec3 ret = incoming - 2 * glm::dot(incoming, normal) * normal;
  return ret;
}

__device__ bool rayTriangleIntersection(const Ray& ray, const Triangle& triangle, float& t)
{
  /* Möller-Trumbore algorithm
   * https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
   */

  const glm::vec3& vertex0 = triangle.vertices[0].p;
  const glm::vec3& vertex1 = triangle.vertices[1].p;
  const glm::vec3& vertex2 = triangle.vertices[2].p;
  glm::vec3 edge1, edge2, h, s, q;
  float a,f,u,v;
  edge1 = vertex1 - vertex0;
  edge2 = vertex2 - vertex0;

  h = glm::cross(ray.direction, edge2);
  a = glm::dot(edge1, h);

  if (a > -EPSILON && a < EPSILON)
    return false;

  f = 1 / a;
  s = ray.origin - vertex0;
  u = f * glm::dot(s, h);

  if (u < 0.f || u > 1.0f)
    return false;

  q = glm::cross(s, edge1);
  v = f * glm::dot(ray.direction, q);

  if (v < 0.0 || u + v > 1.0)
    return false;

  t = f * glm::dot(edge2, q);

  if (t > EPSILON)
  {
    return true;
  }
  else
    return false;
}


__device__ const Material* getMaterial(const int triIdx, const MeshDescriptor* meshes, const int nMeshes)
{
  for (int i = 0; i < nMeshes; ++i)
  {
    const MeshDescriptor& mesh = meshes[i];

    if (mesh.start <= triIdx && triIdx < mesh.start + mesh.nTriangles)
      return &mesh.material;
  }

  return nullptr;
}

// This is where we traverse the acceleration structure later...
__device__ RaycastResult rayCast(const Ray& ray, const Triangle* triangles, const unsigned int nTriangles)
{
  float tMin = BIGT;
  int minTriIdx = -1;

  for (int i = 0; i < nTriangles; ++i)
  {
    float t = 0;

    if (rayTriangleIntersection(ray, triangles[i], t))
    {
      if(t < tMin)
      {
        tMin = t;
        minTriIdx = i;
      }
    }
  }

  if (minTriIdx == -1)
    return RaycastResult();

  glm::fvec3 hitPoint = ray.origin + glm::normalize(ray.direction) * tMin;
  glm::fvec2 uv(0.f);

  return RaycastResult(minTriIdx, tMin, uv, hitPoint);
}


template<typename curandState>
__device__ glm::vec3 areaLightShading(const Light& light, const RaycastResult& result, const Triangle* triangles, const unsigned int nTriangles, curandState& curandState1, curandState& curandState2, const unsigned int supersampling)
{
  const Triangle& hitTriangle = triangles[result.triangleIdx];

  glm::vec3 lightSamplePoint;
  float pdf;

  glm::vec3 brightness(0.f);

  for (unsigned int i = 0; i < supersampling; ++i) // Maybe utilize dynamic parallelism here?
  {
    light.sample(pdf, lightSamplePoint, curandState1, curandState2);

    glm::vec3 shadowRayOrigin = result.point + hitTriangle.normal() * EPSILON;
    glm::vec3 shadowRayDir = lightSamplePoint - shadowRayOrigin;

    float maxT = glm::length(shadowRayDir); // Distance to the light

    Ray shadowRay(shadowRayOrigin, glm::normalize(shadowRayDir));

    RaycastResult shadowResult = rayCast(shadowRay, triangles, nTriangles);

    if ((shadowResult && shadowResult.t >= maxT - EPSILON) || !shadowResult)
    {
      const float cosOmega = glm::clamp(glm::dot(glm::normalize(shadowRayDir), hitTriangle.normal()), 0.f, 1.f);
      const float cosL = glm::clamp(glm::dot(-glm::normalize(shadowRayDir), light.getNormal()), 0.f, 1.f);

      brightness += 1.0f / (glm::dot(shadowRayDir, shadowRayDir) * pdf) * light.getEmission() * cosOmega * cosL;
    }
  }

  brightness /= supersampling;

  return brightness;
}

__device__ glm::fvec3 rayTrace(const Ray& ray, const Triangle* triangles, const int nTriangles, const Camera camera, const MeshDescriptor* meshDescriptors, const int nMeshes, const Light light, curandState_t& curandState1, curandState_t& curandState2, const int recursions)
{
  if (recursions == 0)
    return glm::fvec3(0.f);

  RaycastResult result = rayCast(ray, triangles, nTriangles);

  if (!result)
    return glm::fvec3(0.f);

  glm::fvec3 color(0.f);

  const Triangle& hitTriangle = triangles[result.triangleIdx];

  const Material* material = getMaterial(result.triangleIdx, meshDescriptors, nMeshes);

  color = material->colorAmbient * 0.25f; // Ambient lightning
  color += material->colorDiffuse / glm::pi<float>() * areaLightShading(light, result, triangles, nTriangles, curandState1, curandState2, SHADOWSAMPLING);

  if (glm::length(material->colorSpecular) > 0.0f) {

    glm::fvec3 reflRayOrigin = result.point + hitTriangle.normal() * EPSILON;
    glm::fvec3 reflRayDir = mirrorDirection(hitTriangle.normal(), ray.direction);

    Ray reflRay = Ray(reflRayOrigin, reflRayDir);
    color += material->colorSpecular * rayTrace(reflRay, triangles, nTriangles, camera, meshDescriptors, nMeshes, light, curandState1, curandState2, recursions - 1);
  }

  return color;
}

template<typename curandState>
__global__ void initRand(const int /*seed*/, curandState* const curandStateDevPtr, const glm::ivec2 size)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= size.x || y >= size.y)
    return;

  curandState localState;
  curand_init(x + y*size.x, 0, 0, &localState);
  curandStateDevPtr[x + y * size.x] = localState;
}

template<typename T>
__device__ void writeToCanvas(const unsigned int x, const unsigned int y, const cudaSurfaceObject_t& surfaceObj, const glm::ivec2& canvasSize, T& data)
{
  float4 out = make_float4(data.x, data.y, data.z, 1.f);
  surf2Dwrite(out, surfaceObj, (canvasSize.x - 1 - x) * sizeof(out), y);
  return;
}

__global__ void cudaRender(const cudaSurfaceObject_t canvas, const glm::ivec2 canvasSize, const Triangle* triangles, const int nTriangles, const Camera camera, const MeshDescriptor* meshDescriptors, const int nMeshes, const Light light, curandState_t* curandStateDevXPtr, curandState_t* curandStateDevYPtr)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= canvasSize.x || y >= canvasSize.y)
    return;

  glm::vec2 nic = camera.normalizedImageCoordinateFromPixelCoordinate(glm::ivec2(x, y), canvasSize);

  const float ar = (float) canvasSize.x/canvasSize.y;

  Ray ray = camera.generateRay(nic, ar);

  glm::fvec3 color = rayTrace(ray, triangles, nTriangles, camera, meshDescriptors, nMeshes, light, curandStateDevXPtr[x + WINDOW_MAXWIDTH * y], curandStateDevYPtr[x + WINDOW_MAXWIDTH * y], RECURSIONS);

  writeToCanvas(x, y, canvas, canvasSize, color);

  return;
}


CudaRenderer::CudaRenderer() : curandStateDevVecX(), curandStateDevVecY()
{
  unsigned int cudaDeviceCount = 0;
  int cudaDevices[8];
  unsigned int cudaDevicesCount = 8;

	cudaGLGetDevices(&cudaDeviceCount, cudaDevices, cudaDevicesCount, cudaGLDeviceListCurrentFrame);

  if (cudaDeviceCount < 1)
  {
     std::cout << "No CUDA devices found" << std::endl;
     throw std::runtime_error("No CUDA devices available");
  }

  CUDA_CHECK(cudaSetDevice(cudaDevices[0]));

  curandStateDevVecX.resize(WINDOW_MAXWIDTH * WINDOW_MAXHEIGHT);
  curandStateDevVecY.resize(WINDOW_MAXWIDTH * WINDOW_MAXHEIGHT);
  curandState_t* curandStateDevXRaw = thrust::raw_pointer_cast(&curandStateDevVecX[0]);
  curandState_t* curandStateDevYRaw = thrust::raw_pointer_cast(&curandStateDevVecY[0]);

  dim3 block(BLOCKWIDTH, BLOCKWIDTH);
  dim3 grid( (WINDOW_MAXWIDTH + block.x - 1) / block.x, (WINDOW_MAXHEIGHT + block.y - 1) / block.y);
  initRand<<<grid, block>>>(0, curandStateDevXRaw, glm::ivec2(WINDOW_MAXWIDTH, WINDOW_MAXHEIGHT));
  initRand<<<grid, block>>>(5, curandStateDevYRaw, glm::ivec2(WINDOW_MAXWIDTH, WINDOW_MAXHEIGHT));
  CUDA_CHECK(cudaDeviceSynchronize()); // Would need to wait anyway when initializing models etc.
}

CudaRenderer::~CudaRenderer()
{

}


void CudaRenderer::renderToCanvas(GLCanvas& canvas, const Camera& camera, GLModel& model, GLLight& light)
{
  if (model.getNTriangles() == 0)
    return;

  glm::ivec2 canvasSize = canvas.getSize();

  curandState_t* curandStateDevXRaw = thrust::raw_pointer_cast(&curandStateDevVecX[0]);
  curandState_t* curandStateDevYRaw = thrust::raw_pointer_cast(&curandStateDevVecY[0]);

  auto surfaceObj = canvas.getCudaMappedSurfaceObject();
  Triangle* devTriangles = model.cudaGetMappedTrianglePtr();

  int meshes = model.getNMeshes();

  dim3 block(BLOCKWIDTH, BLOCKWIDTH);
  dim3 grid( (canvasSize.x+ block.x - 1) / block.x, (canvasSize.y + block.y - 1) / block.y);

  cudaRender<<<grid, block>>>(surfaceObj, canvasSize, devTriangles, model.getNTriangles(), camera, model.cudaGetMappedMeshDescriptorPtr(), meshes, light.getLight(), curandStateDevXRaw, curandStateDevYRaw);
  CUDA_CHECK(cudaDeviceSynchronize());


  model.cudaUnmapTrianglePtr();
  canvas.cudaUnmap();
}



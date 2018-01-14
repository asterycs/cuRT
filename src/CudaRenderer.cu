#include "CudaRenderer.hpp"

#include <GL/glew.h>
#include <GL/gl.h>

#include <glm/gtx/component_wise.hpp>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include "Utils.hpp"
#include "Triangle.hpp"


#define BLOCKWIDTH 8
#define INTERSECT_EPSILON 0.0000001f
#define OFFSET_EPSILON 0.00001f
#define BIGT 99999.f
#define SHADOWSAMPLING 8
#define SECONDARY_RAYS 3
#define AIR_INDEX 1.f

#define REFLECTIVE_BIT 0x80000000
#define REFRACTIVE_BIT 0x40000000
#define INSIDE_BIT 0x20000000

#define LEFT_HIT_BIT 0x80000000
#define RIGHT_HIT_BIT 0x40000000

#define PATH_TRACE_BOUNCES 6

inline __device__ float fresnelReflectioncoefficient(const float sin2t, const float cosi, const float idx1, const float idx2)
{
  const float cost = sqrt(1 - sin2t);

  float Rs = (idx1 * cosi - idx2 * cost) / (idx1 * cosi + idx2 * cost);
  Rs = Rs * Rs;

  float Rp = (idx2 * cosi - idx1 * cost) / (idx2 * cosi + idx1 * cost);
  Rp = Rp * Rp;

  return (Rs + Rp) * 0.5f;
}

__device__ glm::fmat3 getBasis(const glm::fvec3 n) {

  glm::fmat3 R;

  glm::fvec3 Q = n;
  const glm::fvec3 absq = glm::abs(Q);
  float absqmin = glm::compMin(absq);

  for (int i = 0; i < 3; ++i) {
    if (absq[i] == absqmin) {
      Q[i] = 1;
      break;
    }
  }

  glm::fvec3 T = glm::normalize(glm::cross(Q, n));
  glm::fvec3 B = glm::normalize(glm::cross(n, T));

  R[0] = T;
  R[1] = B;
  R[2] = n;

  return R;
}

__device__ bool bboxIntersect(const AABB box, const glm::fvec3 origin, const glm::fvec3 inverseDirection, float& t)
{
  glm::fvec3 tmin(-BIGT), tmax(BIGT);

  const glm::fvec3 tdmin = (box.min - origin) * inverseDirection;
  const glm::fvec3 tdmax = (box.max - origin) * inverseDirection;

  tmin = glm::min(tdmin, tdmax);
  tmax = glm::max(tdmin, tdmax);

  const float tmind = glm::compMax(tmin);
  const float tmaxd = glm::compMin(tmax);

  t = fminf(tmind, tmaxd);

  return tmaxd >= tmind && !(tmaxd < 0.f && tmind < 0.f);
}

__device__ void debug_fvec3(const glm::fvec3 v)
{
  printf("%f %f %f\n", v.x, v.y, v.z);
}

inline __device__ glm::fvec3 reflectionDirection(const glm::vec3 normal, const glm::vec3 incoming) {

  const float cosT = glm::dot(incoming, normal);

  return incoming - 2 * cosT * normal;
}

inline __device__ glm::fvec3 refractionDirection(const float cosInAng, const float sin2t, const glm::vec3 normal, const glm::vec3 incoming, const float index1, const float index2)
{
    return index1 / index2 * incoming + (index1 / index2 * cosInAng - sqrt(1 - sin2t)) * normal;
}

__device__ bool rayTriangleIntersection(const Ray ray, const Triangle& triangle, float& t, glm::fvec2& uv)
{
  /* Möller-Trumbore algorithm
   * https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
   */

  // TODO: Experiment with __ldg

  const glm::vec3 vertex0 = triangle.vertices[0].p;

  const glm::fvec3 edge1 = triangle.vertices[1].p - vertex0;
  const glm::fvec3 edge2 = triangle.vertices[2].p - vertex0;

  const glm::fvec3 h = glm::cross(ray.direction, edge2);
  const float a = glm::dot(edge1, h);

  if (a > -INTERSECT_EPSILON && a < INTERSECT_EPSILON)
    return false;

  const float f = __fdividef(1.f, a);
  const glm::fvec3 s = ray.origin - vertex0;
  const float u = f * glm::dot(s, h);

  if (u < 0.f || u > 1.0f)
    return false;

  const glm::fvec3 q = glm::cross(s, edge1);
  const float v = f * glm::dot(ray.direction, q);

  if (v < 0.0 || u + v > 1.0)
    return false;

  t = f * glm::dot(edge2, q);

  if (t > INTERSECT_EPSILON)
  {
    uv = glm::fvec2(u, v);
    return true;
  }
  else
    return false;
}

enum HitType
{
    ANY,
    CLOSEST
};

template <bool debug, const HitType hitType>
__device__
RaycastResult rayCast(const Ray ray, const Node* bvh, const Triangle* triangles, const float maxT)
{
  float tMin = maxT;
  int minTriIdx = -1;
  glm::fvec2 minUV;
  RaycastResult result;
  const glm::fvec3 inverseDirection = glm::fvec3(1.f) / ray.direction;

  int ptr = 0;
  unsigned int stack[8] { 0 };
  int i = -1;
  float t = 0;
  glm::fvec2 uv;
  bool getNextNode = true;

  while (ptr >= 0)
  {
    unsigned int currentNodeIdx = stack[ptr];
    Node currentNode = bvh[currentNodeIdx];


    if (currentNode.rightIndex == -1)
    {
      getNextNode = false;

      if (debug)
      {
        const AABB b = currentNode.bbox;
        printf("\nHit bbox %d:\n", currentNodeIdx);
        printf("min: %f %f %f\n", b.min[0], b.min[1], b.min[2]);
        printf("max: %f %f %f\n", b.max[0], b.max[1], b.max[2]);
        printf("StartIdx: %d, endIdx: %d, nTris: %d\n\n", currentNode.startTri, currentNode.startTri + currentNode.nTri, currentNode.nTri);
      }

      if (i >= currentNode.startTri && i < currentNode.startTri + currentNode.nTri)
      {
        if (rayTriangleIntersection(ray, triangles[i], t, uv))
        {
          if (debug)
            printf("Hit triangle %d\n", i);

          if(t < tMin)
          {
            tMin = t;
            minTriIdx = i;
            minUV = uv;

            if (hitType == HitType::ANY)
              break;
          }
        }

        ++i;

        if (i >= currentNode.startTri + currentNode.nTri)
          getNextNode = true;

      }else
      {
        i = currentNode.startTri;
      }

    }else
    {
      const AABB leftBox = bvh[stack[ptr] + 1].bbox;
      const AABB rightBox = bvh[currentNode.rightIndex].bbox;

      float leftt, rightt;

      unsigned int hitMask = bboxIntersect(leftBox, ray.origin, inverseDirection, leftt) ? LEFT_HIT_BIT : 0x00;
      hitMask = bboxIntersect(rightBox, ray.origin, inverseDirection, rightt) ? hitMask | RIGHT_HIT_BIT : hitMask;

      // TODO: Push closer one last, don't intersect if closest hit is closer than box
      if ((hitMask & LEFT_HIT_BIT) != 0x00 && leftt < tMin)
      {
        stack[ptr] = currentNodeIdx + 1;
        ++ptr;
      }

      if ((hitMask & RIGHT_HIT_BIT) != 0x00 && rightt < tMin)
      {
        stack[ptr] = currentNode.rightIndex;
        ++ptr;
      }
    }

    if (getNextNode)
    {
      --ptr;
      i = -1;
    }

  }

  if (minTriIdx == -1)
    return result;

  result.point = ray.origin + ray.direction * tMin;
  result.t = tMin;
  result.triangleIdx = minTriIdx;
  result.uv = minUV;

  if (debug)
    printf("///////////////////\n\n");

  return result;
}


template<unsigned int samples, typename curandState>
__device__ glm::fvec3 areaLightShading(const glm::fvec3 interpolatedNormal, const Light& light, const Node* bvh, const RaycastResult& result, const Triangle* triangles, curandState& curandState1, curandState& curandState2)
{
  glm::fvec3 brightness(0.f);

  //if (!light.isEnabled()) // Surprisingly slow
  //  return brightness;

  const glm::fvec3 shadowRayOrigin = result.point + interpolatedNormal * OFFSET_EPSILON;

  glm::fvec3 lightSamplePoint;
  float pdf;

  const glm::fvec3 emission = light.getEmission();

  // TODO: Unroll using templates
  for (unsigned int i = 0; i < samples; ++i)
  {
    light.sample(pdf, lightSamplePoint, curandState1, curandState2);

    const glm::fvec3 shadowRayDir = lightSamplePoint - shadowRayOrigin;

    const float maxT = glm::length(shadowRayDir); // Distance to the light
    const glm::fvec3 shadowRayDirNormalized = shadowRayDir / maxT;

    const Ray shadowRay(shadowRayOrigin, shadowRayDirNormalized);

    const RaycastResult shadowResult = rayCast<false, HitType::ANY>(shadowRay, bvh, triangles, maxT);

    if ((shadowResult && shadowResult.t >= maxT + OFFSET_EPSILON) || !shadowResult)
    {
      const float cosOmega = __saturatef(glm::dot(shadowRayDirNormalized, interpolatedNormal));
      const float cosL = __saturatef(glm::dot(-shadowRayDirNormalized, light.getNormal()));

      brightness += __fdividef(1.f, (maxT * maxT * pdf)) * emission * cosL * cosOmega;
    }
  }

  brightness /= samples;

  return brightness;
}

__device__ inline constexpr unsigned int cpow(const unsigned int base, const unsigned int exponent)
{
    return (exponent == 0) ? 1 : (base * cpow(base, exponent - 1));
}

struct RaycastTask
{
  Ray outRay;
  unsigned short levelsLeft;
  glm::fvec3 filter;
};

template <bool debug, typename curandStateType>
__device__ glm::fvec3 rayTrace(\
    const Node* bvh, \
    const Ray& ray, \
    const Triangle* triangles, \
    const Camera camera, \
    const Material* materials, \
    const unsigned int* triangleMaterialIds, \
    const Light light, \
    curandStateType& curandState1, \
    curandStateType& curandState2, \
    glm::fvec3* hitPoints = nullptr)
{
  constexpr unsigned int stackSize = cpow(2, SECONDARY_RAYS);
  RaycastTask stack[stackSize];
  glm::fvec3 color(0.f);
  int stackPtr = 0;
  int posPtr = 0; // Probably optimized away when not used

  // Primary ray
  stack[stackPtr].outRay = ray;
  stack[stackPtr].levelsLeft = SECONDARY_RAYS;
  stack[stackPtr].filter = glm::fvec3(1.f);
  ++stackPtr;

  while (stackPtr > 0)
  {
    --stackPtr;

    const RaycastTask currentTask = stack[stackPtr];
    const RaycastResult result = rayCast<false, HitType::CLOSEST>(currentTask.outRay, bvh, triangles, BIGT);

    if (!result)
      continue;

    if (debug)
    {
      hitPoints[posPtr++] = currentTask.outRay.origin;
      hitPoints[posPtr++] = result.point;
    }
    
    const Triangle triangle = triangles[result.triangleIdx];
    const Material material = materials[triangleMaterialIds[result.triangleIdx]];
    glm::fvec3 interpolatedNormal = triangle.normal(result.uv);

    unsigned int mask = INSIDE_BIT;

    if (glm::dot(interpolatedNormal, currentTask.outRay.direction) > 0.f)
      interpolatedNormal = -interpolatedNormal;  // We are inside an object. Flip the normal.
    else
      mask = 0x00000000; // We are outside. Unset bit.

    color += currentTask.filter * material.colorAmbient * 0.25f;

    const glm::fvec3 brightness = areaLightShading<SHADOWSAMPLING>(interpolatedNormal, light, bvh, result, triangles, curandState1, curandState2);
    color += currentTask.filter * material.colorDiffuse / glm::pi<float>() * brightness;

    if (material.shadingMode == material.GORAUD)
    {
      continue;
    }

    // Phong's specular highlight
    if ((mask & INSIDE_BIT) == 0x00 && material.shadingMode == material.PHONG)
    {
      const glm::fvec3 rm = reflectionDirection(interpolatedNormal, glm::normalize(light.getPosition() - result.point));
      color += material.colorSpecular * powf(__saturatef(glm::dot(rm, currentTask.outRay.direction)), material.shininess);
    }

    if (material.shadingMode == material.FRESNEL)
    {

      if (currentTask.levelsLeft == 0)
        continue;

      RaycastTask newTask; // Used twice for pushing

      mask = (material.colorSpecular.x != 0.f ||
          material.colorSpecular.y != 0.f ||
          material.colorSpecular.z != 0.f) ? REFLECTIVE_BIT | mask : mask;

      mask = (material.colorTransparent.x != 0.f ||
          material.colorTransparent.y != 0.f ||
          material.colorTransparent.z != 0.f) ? REFRACTIVE_BIT | mask : mask;

      float R = 1.f;

      if ((mask & REFRACTIVE_BIT) != 0x00) // Refractive
      {
        float idx1 = AIR_INDEX;
        float idx2 = material.refrIdx;

        float rat;

        if ((mask & INSIDE_BIT) != 0x00) // inside
          rat = __fdividef(idx1, idx2);
        else
          rat = __fdividef(idx2, idx1);

        // Something's not right here...

        // Transmittance and reflection according to fresnel
        const float cosi = fabsf(glm::dot(currentTask.outRay.direction, interpolatedNormal));

        if (sinf(acosf(cosi)) <= rat) // Check for total internal reflection
        {
          const float sin2t = fabs((idx1 / idx2) * (idx1 / idx2) * (1 - cosi * cosi));

          R = fresnelReflectioncoefficient(sin2t, cosi, idx1, idx2);

          const glm::fvec3 transOrig = result.point - interpolatedNormal * OFFSET_EPSILON;
          const glm::fvec3 transDir = refractionDirection(cosi, sin2t, interpolatedNormal, currentTask.outRay.direction, idx1, idx2);

          newTask.outRay = Ray(transOrig, transDir);
          newTask.levelsLeft = currentTask.levelsLeft - 1;
          newTask.filter = currentTask.filter * material.colorTransparent * (1 - R);
          stack[stackPtr] = newTask;
          ++stackPtr;
        }
      }

      if ((mask & REFLECTIVE_BIT) != 0x00) // Reflective
      {
        const glm::fvec3 reflOrig = result.point + interpolatedNormal * OFFSET_EPSILON;
        const glm::fvec3 reflDir = reflectionDirection(interpolatedNormal, currentTask.outRay.direction);

        newTask.outRay = Ray(reflOrig, reflDir);
        newTask.levelsLeft = currentTask.levelsLeft - 1;
        newTask.filter = currentTask.filter * material.colorSpecular * R;
        stack[stackPtr] = newTask;
        ++stackPtr;
      }
    }

  }

  return color;
}

template <const bool debug, typename curandStateType>
__device__ glm::fvec3 pathTrace(\
    const Node* bvh, \
    const Ray& ray, \
    const Triangle* triangles, \
    const Camera camera, \
    const Material* materials, \
    const unsigned int* triangleMaterialIds, \
    const Light light, \
    curandStateType& curandState1, \
    curandStateType& curandState2, \
    glm::fvec3* hitPoints = nullptr)
{
  unsigned int posPtr = 0;

  Ray currentRay = ray;
  glm::fvec3 color(0.f, 0.f, 0.f);
  glm::fvec3 throughput(1.f, 1.f, 1.f);

  float p = 1.0f;
  bool roulette = false;

  unsigned int bounces = PATH_TRACE_BOUNCES;
  bool terminate = false;
  unsigned int currentBounce = 0;

  do
  {
    const RaycastResult result = rayCast<debug, HitType::CLOSEST>(currentRay, bvh, triangles, BIGT);

    if (!result)
      return color;

    if (debug)
    {
      hitPoints[posPtr++] = currentRay.origin;
      hitPoints[posPtr++] = result.point;
    }

    const Triangle triangle = triangles[result.triangleIdx];
    const Material material = materials[triangleMaterialIds[result.triangleIdx]];
    glm::fvec3 interpolatedNormal = triangle.normal(result.uv);

    unsigned int mask = INSIDE_BIT;

    if (glm::dot(interpolatedNormal, currentRay.direction) > 0.f)
      interpolatedNormal = -interpolatedNormal;  // We are inside an object. Flip the normal.
    else
      mask = 0x00; // We are outside. Unset bit.

    color += throughput * material.colorAmbient * 0.25f;
    const glm::fvec3 brightness = areaLightShading<1>(interpolatedNormal, light, bvh, result, triangles, curandState1, curandState2);
    color += throughput * material.colorDiffuse / (glm::pi<float>() * p) * brightness;

    // Phong's specular highlight
    if ((mask & INSIDE_BIT) == 0x00 && material.shadingMode == material.PHONG)
    {
      const glm::fvec3 rm = reflectionDirection(interpolatedNormal, glm::normalize(light.getPosition() - result.point));
      color += material.colorSpecular * powf(__saturatef(glm::dot(rm, currentRay.direction)), material.shininess);
    }


    glm::fvec3 newDir, newOrig;
    glm::fmat3 B;

    if (material.shadingMode == material.FRESNEL)
    {
      mask = (material.colorSpecular.x != 0.f ||
          material.colorSpecular.y != 0.f ||
          material.colorSpecular.z != 0.f) ? REFLECTIVE_BIT | mask : mask;

      mask = (material.colorTransparent.x != 0.f ||
          material.colorTransparent.y != 0.f ||
          material.colorTransparent.z != 0.f) ? REFRACTIVE_BIT | mask : mask;

      float rP = 1.f; // Probability for reflection to occur. Depends on the strength of the specular and transparent colors.

      float R = 1.f; // Fresnel reflection coefficient
      float cosi, sin2t, idx1, idx2;

      if ((mask & REFRACTIVE_BIT) != 0x00)
      {
        float rLen = glm::length(material.colorSpecular);
        float tLen = glm::length(material.colorTransparent);

        rP = rLen / (rLen + tLen);

        idx1 = AIR_INDEX;
        idx2 = material.refrIdx;

        float rat;

        if ((mask & INSIDE_BIT) != 0x00) // inside
          rat = __fdividef(idx1, idx2);
        else
          rat = __fdividef(idx2, idx1);

        cosi = fabsf(glm::dot(currentRay.direction, interpolatedNormal));

        if (sinf(acosf(cosi)) <= rat) // Check for total internal reflection
        {
          sin2t = fabs((idx1 / idx2) * (idx1 / idx2) * (1 - cosi * cosi));
          R = fresnelReflectioncoefficient(sin2t, cosi, idx1, idx2);
        }
      }

      rP *= R;

      rP = rP / (rP + (1.f - rP) * (1.f - R));

      bool refl = curand_uniform(&curandState1) < rP;

      if (refl)
      {
        newDir = reflectionDirection(interpolatedNormal, currentRay.direction);
        newOrig = result.point + interpolatedNormal * OFFSET_EPSILON;
        throughput *= material.colorSpecular;
      }
      else
      {
        newDir = refractionDirection(cosi, sin2t, interpolatedNormal, currentRay.direction, idx1, idx2);
        newOrig = result.point - interpolatedNormal * OFFSET_EPSILON;
        throughput *= material.colorTransparent;
      }

    }
    else // Diffuse
    {
      B = getBasis(interpolatedNormal);

      do {
        newDir = glm::fvec3(curand_uniform(&curandState1) * 2.0f - 1.0f, curand_uniform(&curandState1) * 2.0f - 1.0f, 0.f);
      } while ((newDir.x * newDir.x + newDir.y * newDir.y) >= 1);

      newDir.z = glm::sqrt(1 - newDir.x * newDir.x - newDir.y * newDir.y);
      newDir = B * newDir;
      newDir = glm::normalize(newDir);

      newOrig = result.point + OFFSET_EPSILON * interpolatedNormal;

      p *= glm::dot(newDir, interpolatedNormal) * (1.f / glm::pi<float>());
      throughput *= material.colorDiffuse / glm::pi<float>() * glm::dot(newDir, interpolatedNormal);
    }

    currentRay = Ray(newOrig, newDir);

    if (currentBounce < bounces)
    {
      ++currentBounce;
    }
    else if (roulette)
    {
      ++currentBounce;
      p *= 0.8f; // Continuation probability
      terminate = curand_uniform(&curandState1) < 0.2f;
    }
    else
      terminate = true;

  } while (!terminate);


  return color;
}

template <typename curandState>
__global__ void initRand(const int seed, curandState* const curandStateDevPtr, const glm::ivec2 size)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= size.x || y >= size.y)
    return;

  curandState localState;
  curand_init(seed, x + y*size.x, 0, &localState);
  curandStateDevPtr[x + y * size.x] = localState;
}

__global__ void initRand(curandDirectionVectors64_t* sobolDirectionVectors, unsigned long long* sobolScrambleConstants, curandStateScrambledSobol64* state, const glm::ivec2 size)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  
  if (x >= size.x || y >= size.y)
    return;
    
  const unsigned int scrIdx = x + size.x * y;
  const unsigned int dirIdx = (x + size.x * y) % 10000;

  curandDirectionVectors64_t* dir = &sobolDirectionVectors[dirIdx];
  unsigned long long scr = sobolScrambleConstants[scrIdx];
  curandStateScrambledSobol64 localState;
    
  curand_init(*dir, scr, 0, &localState);

  state[x + size.x * y] = localState;
}

__device__ void writeToCanvas(const unsigned int x, const unsigned int y, const cudaSurfaceObject_t& surfaceObj, const glm::ivec2 canvasSize, const glm::vec3 data)
{
  const float4 out = make_float4(data.x, data.y, data.z, 1.f);
  surf2Dwrite(out, surfaceObj, (canvasSize.x - 1 - x) * sizeof(out), y);
  return;
}

__device__ glm::fvec3 readFromCanvas(const unsigned int x, const unsigned int y, const cudaSurfaceObject_t& surfaceObj, const glm::ivec2 canvasSize)
{
  float4 in;
  surf2Dread(&in, surfaceObj, (canvasSize.x - 1 - x) * sizeof(in), y);

  const glm::fvec3 ret(in.x, in.y, in.z);

  return ret;
}

template <typename curandStateType>
__global__ void
cudaDebugRayTrace(\
    const glm::ivec2 pixelPos, \
    glm::fvec3* devPosPtr, \
    const glm::ivec2 size, \
    const Triangle* triangles, \
    const Camera camera, \
    const Material* materials, \
    const unsigned int* triangleMaterialIds, \
    const Light light, \
    curandStateType* curandStateDevXPtr, \
    curandStateType* curandStateDevYPtr, \
    const Node* bvh)
{
  const glm::fvec2 nic = camera.normalizedImageCoordinateFromPixelCoordinate(pixelPos.x, pixelPos.y, size);
  const float ar = (float) size.x / size.y;
  const Ray ray = camera.generateRay(nic, ar);

  (void) rayTrace<true>(\
      bvh,
      ray, \
      triangles, \
      camera, \
      materials, \
      triangleMaterialIds, \
      light, \
      curandStateDevXPtr[pixelPos.x + size.x * pixelPos.y], \
      curandStateDevYPtr[pixelPos.x + size.x * pixelPos.y], \
      devPosPtr);

  return;
}

template <typename curandStateType>
__global__ void
cudaDebugPathTrace(
    const glm::ivec2 pixelPos,
    glm::fvec3* devPosPtr,
    const glm::ivec2 size,
    const Triangle* triangles,
    const Camera camera,
    const Material* materials,
    const unsigned int* triangleMaterialIds,
    const Light light,
    curandStateType* curandStateDevXPtr,
    curandStateType* curandStateDevYPtr,
    const Node* bvh)
{
  const glm::fvec2 nic = camera.normalizedImageCoordinateFromPixelCoordinate(pixelPos.x, pixelPos.y, size);
  const float ar = (float) size.x / size.y;
  const Ray ray = camera.generateRay(nic, ar);

  (void) pathTrace<true>(
      bvh,
      ray,
      triangles,
      camera,
      materials,
      triangleMaterialIds,
      light,
      curandStateDevXPtr[pixelPos.x + size.x * pixelPos.y],
      curandStateDevYPtr[pixelPos.x + size.x * pixelPos.y],
      devPosPtr);

  return;
}

template <typename curandStateType>
__global__ void
__launch_bounds__(BLOCKWIDTH * BLOCKWIDTH, 24)
rayTraceKernel(\
    const cudaSurfaceObject_t canvas, \
    const glm::ivec2 canvasSize, \
    const Triangle* triangles, \
    const Camera camera, \
    const Material* materials, \
    const unsigned int* triangleMaterialIds, \
    const Light light, \
    curandStateType* curandStateDevXPtr, \
    curandStateType* curandStateDevYPtr, \
    const Node* bvh)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= canvasSize.x || y >= canvasSize.y)
    return;

  glm::vec2 nic = camera.normalizedImageCoordinateFromPixelCoordinate(x, y, canvasSize);

  Ray ray = camera.generateRay(nic, (float) canvasSize.x/canvasSize.y);

  curandStateType state1 = curandStateDevXPtr[x + y * canvasSize.x];
  curandStateType state2 = curandStateDevYPtr[x + y * canvasSize.x];

  glm::fvec3 color = rayTrace<false>(\
      bvh,
      ray, \
      triangles, \
      camera, \
      materials, \
      triangleMaterialIds, \
      light, \
      state1, \
      state2);

  curandStateDevXPtr[x + y * canvasSize.x] = state1;
  curandStateDevYPtr[x + y * canvasSize.x] = state2;

  writeToCanvas(x, y, canvas, canvasSize, color);

  return;
}

template <typename curandStateType>
__global__ void
pathTraceKernel(
    const unsigned int path,
    const cudaSurfaceObject_t canvas,
    const glm::ivec2 canvasSize,
    const Triangle* triangles,
    const Camera camera,
    const Material* materials,
    const unsigned int* triangleMaterialIds,
    const Light light,
    curandStateType* curandStateDevXPtr,
    curandStateType* curandStateDevYPtr,
    const Node* bvh)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= canvasSize.x || y >= canvasSize.y)
    return;

  glm::vec2 nic = camera.normalizedImageCoordinateFromPixelCoordinate(x, y, canvasSize);

  Ray ray = camera.generateRay(nic, (float) canvasSize.x/canvasSize.y);

  curandStateType state1 = curandStateDevXPtr[x + y * canvasSize.x];
  curandStateType state2 = curandStateDevYPtr[x + y * canvasSize.x];

  glm::fvec3 color = pathTrace<false>(\
      bvh,
      ray, \
      triangles, \
      camera, \
      materials, \
      triangleMaterialIds, \
      light, \
      state1, \
      state2);

  curandStateDevXPtr[x + y * canvasSize.x] = state1;
  curandStateDevYPtr[x + y * canvasSize.x] = state2;

  if (path == 1)
  {
    writeToCanvas(x, y, canvas, canvasSize, color);
  }
  else
  {
    const glm::fvec3 oldCol = readFromCanvas(x, y, canvas, canvasSize);
    const glm::fvec3 blend = oldCol * glm::fvec3((float) (path - 1) / path) + glm::fvec3((float) 1 / path) * color;
    writeToCanvas(x, y, canvas, canvasSize, blend);

  }

  return;
}

template <typename curandStateType>
__global__ void
cudaTestRnd(\
    const cudaSurfaceObject_t canvas, \
    const glm::ivec2 canvasSize, \
    curandStateType* curandStateDevXPtr, \
    curandStateType* curandStateDevYPtr)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= canvasSize.x || y >= canvasSize.y)
    return;

  curandStateType localState1 = curandStateDevXPtr[x + y * canvasSize.x];
  curandStateType localState2 = curandStateDevYPtr[x + y * canvasSize.x];

  float r = curand_uniform(&localState1);
  float g = curand_uniform(&localState2);

  curandStateDevXPtr[x + y * canvasSize.x] = localState1;
  curandStateDevYPtr[x + y * canvasSize.x] = localState2;

  writeToCanvas(x, y, canvas, canvasSize, glm::fvec3(r, g, 0.f));

  return;
}

void CudaRenderer::reset()
{
  currentPath = 1;
}

void CudaRenderer::resize(const glm::ivec2& size)
{
  curandStateDevVecX.resize(size.x * size.y);
  curandStateDevVecY.resize(size.x * size.y);
  auto* curandStateDevXRaw = thrust::raw_pointer_cast(&curandStateDevVecX[0]);
  auto* curandStateDevYRaw = thrust::raw_pointer_cast(&curandStateDevVecY[0]);

  dim3 block(BLOCKWIDTH, BLOCKWIDTH);
  dim3 grid( (size.x + block.x - 1) / block.x, (size.y + block.y - 1) / block.y);

  cudaStream_t streams[2];
  CUDA_CHECK(cudaStreamCreate(&streams[0]));
  CUDA_CHECK(cudaStreamCreate(&streams[1]));

#ifdef QUASIRANDOM
  curandDirectionVectors64_t* hostDirectionVectors64;
  unsigned long long int* hostScrambleConstants64;
  
  curandDirectionVectors64_t* devDirectionVectors64;
  unsigned long long int* devScrambleConstants64;
  
  curandGetDirectionVectors64(&hostDirectionVectors64, CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6);
  curandGetScrambleConstants64(&hostScrambleConstants64);
  
  CUDA_CHECK(cudaMalloc((void **)&(devDirectionVectors64),             20000 * sizeof(curandDirectionVectors64_t)));
  CUDA_CHECK(cudaMemcpy(devDirectionVectors64, hostDirectionVectors64, 20000 * sizeof(curandDirectionVectors64_t), cudaMemcpyHostToDevice));
  
  CUDA_CHECK(cudaMalloc((void **)&(devScrambleConstants64),              size.x * size.y * sizeof(unsigned long long int)));
  CUDA_CHECK(cudaMemcpy(devScrambleConstants64, hostScrambleConstants64, size.x * size.y * sizeof(unsigned long long int), cudaMemcpyHostToDevice));
  
  initRand<<<grid, block, 0, streams[0]>>>(devDirectionVectors64, devScrambleConstants64, curandStateDevXRaw, size);
  initRand<<<grid, block, 0, streams[1]>>>(devDirectionVectors64 + 10000, devScrambleConstants64, curandStateDevYRaw, size);
  
  CUDA_CHECK(cudaFree(devDirectionVectors64));
  CUDA_CHECK(cudaFree(devScrambleConstants64));

#else
  initRand<<<grid, block, 0, streams[0]>>>(0, curandStateDevXRaw, size);
  initRand<<<grid, block, 0, streams[1]>>>(5, curandStateDevYRaw, size);
#endif

  CUDA_CHECK(cudaStreamDestroy(streams[0]));
  CUDA_CHECK(cudaStreamDestroy(streams[1]));

  CUDA_CHECK(cudaDeviceSynchronize());
}


CudaRenderer::CudaRenderer() : curandStateDevVecX(), curandStateDevVecY(), lastCamera(), lastSize(), currentPath(1)
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

  resize(glm::ivec2(WWIDTH, WHEIGHT));
}

CudaRenderer::~CudaRenderer()
{

}

void CudaRenderer::pathTraceToCanvas(GLTexture& canvas, const Camera& camera, GLModel& model, GLLight& light)
{
  if (model.getNTriangles() == 0)
    return;

  const glm::ivec2 canvasSize = canvas.getSize();
  const bool diffCamera = std::memcmp(&camera, &lastCamera, sizeof(Camera));
  const bool diffSize = (canvasSize != lastSize);

  if (diffCamera != 0 || diffSize != 0)
  {
    lastCamera = camera;
    lastSize = canvasSize;
    currentPath = 1;
  }

  auto* curandStateDevXRaw = thrust::raw_pointer_cast(&curandStateDevVecX[0]);
  auto* curandStateDevYRaw = thrust::raw_pointer_cast(&curandStateDevVecY[0]);

  auto surfaceObj = canvas.getCudaMappedSurfaceObject();
  const Triangle* devTriangles = model.getMappedCudaTrianglePtr();

  const dim3 block(BLOCKWIDTH, BLOCKWIDTH);
  const dim3 grid( (canvasSize.x+ block.x - 1) / block.x, (canvasSize.y + block.y - 1) / block.y);

  pathTraceKernel<<<grid, block>>>(
      currentPath,
      surfaceObj,
      canvasSize,
      devTriangles,
      camera,
      model.getCudaMaterialsPtr(),
      model.getCudaTriangleMaterialIdsPtr(),
      light.getLight(),
      curandStateDevXRaw,
      curandStateDevYRaw,
      model.getDeviceBVH());

  ++currentPath;

  CUDA_CHECK(cudaDeviceSynchronize());
  model.unmapCudaTrianglePtr();
  canvas.cudaUnmap();
}

void CudaRenderer::rayTraceToCanvas(GLTexture& canvas, const Camera& camera, GLModel& model, GLLight& light)
{
  if (model.getNTriangles() == 0)
    return;

  const glm::ivec2 canvasSize = canvas.getSize();

  auto* curandStateDevXRaw = thrust::raw_pointer_cast(&curandStateDevVecX[0]);
  auto* curandStateDevYRaw = thrust::raw_pointer_cast(&curandStateDevVecY[0]);

  auto surfaceObj = canvas.getCudaMappedSurfaceObject();
  const Triangle* devTriangles = model.getMappedCudaTrianglePtr();

  const dim3 block(BLOCKWIDTH, BLOCKWIDTH);
  const dim3 grid( (canvasSize.x+ block.x - 1) / block.x, (canvasSize.y + block.y - 1) / block.y);
  //dim3 block(BLOCKWIDTH * BLOCKWIDTH);
  //dim3 grid( (canvasSize.x * canvasSize.y + block.x - 1) / block.x);

  rayTraceKernel<<<grid, block>>>(\
      surfaceObj, \
      canvasSize, \
      devTriangles, \
      camera, \
      model.getCudaMaterialsPtr(), \
      model.getCudaTriangleMaterialIdsPtr(), \
      light.getLight(), \
      curandStateDevXRaw, \
      curandStateDevYRaw, \
      model.getDeviceBVH());

  //cudaTestRnd<<<grid, block>>>(surfaceObj, canvasSize, curandStateDevXRaw, curandStateDevYRaw);

  CUDA_CHECK(cudaDeviceSynchronize());
  model.unmapCudaTrianglePtr();
  canvas.cudaUnmap();
}

std::vector<glm::fvec3> CudaRenderer::debugRayTrace(const glm::ivec2 pixelPos, const glm::ivec2 size, const Camera& camera, GLModel& model, GLLight& light)
{
  if (model.getNTriangles() == 0)
    return std::vector<glm::fvec3>();

  auto* curandStateDevXRaw = thrust::raw_pointer_cast(&curandStateDevVecX[0]);
  auto* curandStateDevYRaw = thrust::raw_pointer_cast(&curandStateDevVecY[0]);

  Triangle* devTriangles = model.getMappedCudaTrianglePtr();

  dim3 block(1, 1);
  dim3 grid(1, 1);

  unsigned int secondaryVertices = std::pow(2u, SECONDARY_RAYS) == 1 ? 0 : std::pow(2u, SECONDARY_RAYS) * 2;
  const int nVertices = (2 + secondaryVertices) * SHADOWSAMPLING;

  glm::fvec3* devPosPtr;
  CUDA_CHECK(cudaMalloc((void**) &devPosPtr, nVertices * sizeof(glm::fvec3)));
  CUDA_CHECK(cudaMemset((void*) devPosPtr, 0, nVertices * sizeof(glm::fvec3)));

  cudaDebugRayTrace<<<grid, block>>>(\
      size - pixelPos, \
      devPosPtr, \
      size, \
      devTriangles, \
      camera, \
      model.getCudaMaterialsPtr(), \
      model.getCudaTriangleMaterialIdsPtr(), \
      light.getLight(), \
      curandStateDevXRaw, \
      curandStateDevYRaw, \
      model.getDeviceBVH());

  CUDA_CHECK(cudaDeviceSynchronize());
  model.unmapCudaTrianglePtr();

  std::vector<glm::fvec3> hitPos(nVertices);
  CUDA_CHECK(cudaMemcpy(hitPos.data(), devPosPtr, nVertices * sizeof(glm::fvec3), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(devPosPtr));

  return hitPos;
}

std::vector<glm::fvec3> CudaRenderer::debugPathTrace(const glm::ivec2 pixelPos, const glm::ivec2 size, const Camera& camera, GLModel& model, GLLight& light)
{
  if (model.getNTriangles() == 0)
    return std::vector<glm::fvec3>();

  auto* curandStateDevXRaw = thrust::raw_pointer_cast(&curandStateDevVecX[0]);
  auto* curandStateDevYRaw = thrust::raw_pointer_cast(&curandStateDevVecY[0]);

  Triangle* devTriangles = model.getMappedCudaTrianglePtr();

  dim3 block(1, 1);
  dim3 grid(1, 1);

  const int nVertices = 2 * PATH_TRACE_BOUNCES;

  glm::fvec3* devPosPtr;
  CUDA_CHECK(cudaMalloc((void**) &devPosPtr, nVertices * sizeof(glm::fvec3)));
  CUDA_CHECK(cudaMemset((void*) devPosPtr, 0, nVertices * sizeof(glm::fvec3)));

  cudaDebugPathTrace<<<grid, block>>>(\
      size - pixelPos, \
      devPosPtr, \
      size, \
      devTriangles, \
      camera, \
      model.getCudaMaterialsPtr(), \
      model.getCudaTriangleMaterialIdsPtr(), \
      light.getLight(), \
      curandStateDevXRaw, \
      curandStateDevYRaw, \
      model.getDeviceBVH());

  CUDA_CHECK(cudaDeviceSynchronize());
  model.unmapCudaTrianglePtr();

  std::vector<glm::fvec3> hitPos(nVertices);
  CUDA_CHECK(cudaMemcpy(hitPos.data(), devPosPtr, nVertices * sizeof(glm::fvec3), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(devPosPtr));

  return hitPos;
}




when defined(SomeReal) and not defined(SomeFloat):
  type SomeFloat = SomeReal

import vec, mat, math

type
  Quat*[T] = object
    arr*: array[4, T]

proc `[]`*[T](q : Quat[T], i : int) : T =
  q.arr[i]

proc `[]`*[T](q : var Quat[T], i : int) : var T =
  q.arr[i]

proc `[]=`*[T](q : var Quat[T], i : int, val : T) =
  q.arr[i] = val

iterator items*[T](q: Quat[T]): T =
  for x in q.arr.items:
    yield x

iterator mitems*[T](q: var Quat[T]): var T =
  for x in q.arr.mitems:
    yield x

iterator pairs*[T](q: Quat[T]): tuple[key: int, val: T] =
  for i, x in q.arr.pairs:
    yield (i,x)

iterator mpairs*[T](q: var Quat[T]): tuple[key: int, val: var T] =
  for i, x in q.arr.mpairs:
    yield (i,x)

proc `$`*[T](q : Quat[T]) : string =
  result = "quatf("
  for i, val in q.arr:
    if i != 0:
      result &= ", "
    result &= $val
  result &= ")"

proc x*[T](q : Quat[T]) : T = q.arr[0]
proc y*[T](q : Quat[T]) : T = q.arr[1]
proc z*[T](q : Quat[T]) : T = q.arr[2]
proc w*[T](q : Quat[T]) : T = q.arr[3]
proc x*[T](q : var Quat[T]) : var T = q.arr[0]
proc y*[T](q : var Quat[T]) : var T = q.arr[1]
proc z*[T](q : var Quat[T]) : var T = q.arr[2]
proc w*[T](q : var Quat[T]) : var T = q.arr[3]

proc `x=`*[T](q : var Quat[T]; v: T) : void =
  q[0] = v

proc `y=`*[T](q : var Quat[T]; v: T) : void =
  q[1] = v

proc `z=`*[T](q : var Quat[T]; v: T) : void =
  q[2] = v

proc `w=`*[T](q : var Quat[T]; v: T) : void =
  q[3] = v

proc quat*[T](x,y,z,w : T) : Quat[T] {.inline.} =
  ## warning unlike original glm, this constructor does have ``w`` as
  ## the last argument.  The reason I did this, was because I had a
  ## bug with original glm that took me hours to fix just because glm
  ## uses ``wxyz`` as argument order, but ``xyzw`` internally.
  result.arr = [x,y,z,w]

proc quat*[T](axis: Vec3[T]; angle: T): Quat[T] =
  let s = sin(angle / 2)
  result.x = axis.x * s
  result.y = axis.y * s
  result.z = axis.z * s
  result.w = cos(angle / 2)

proc quat*[T](m: Mat3[T]): Quat[T] =
  ## mat needs to be rotation matrix (orthogonal, det(mat) = 1)

  let fourXSquaredMinus1 = m[0][0] - m[1][1] - m[2][2];
  let fourYSquaredMinus1 = m[1][1] - m[0][0] - m[2][2];
  let fourZSquaredMinus1 = m[2][2] - m[0][0] - m[1][1];
  let fourWSquaredMinus1 = m[0][0] + m[1][1] + m[2][2];

  var biggestIndex: 0..3 = 0;
  var fourBiggestSquaredMinus1 = fourWSquaredMinus1;
  if fourXSquaredMinus1 > fourBiggestSquaredMinus1:
      fourBiggestSquaredMinus1 = fourXSquaredMinus1;
      biggestIndex = 1;
  if fourYSquaredMinus1 > fourBiggestSquaredMinus1:
      fourBiggestSquaredMinus1 = fourYSquaredMinus1;
      biggestIndex = 2;
  if fourZSquaredMinus1 > fourBiggestSquaredMinus1:
      fourBiggestSquaredMinus1 = fourZSquaredMinus1;
      biggestIndex = 3;

  let biggestVal = sqrt(fourBiggestSquaredMinus1 + T(1)) * T(0.5);
  let mult = T(0.25) / biggestVal;

  case biggestIndex
  of 0:
    result.w = biggestVal;
    result.x = (m[1][2] - m[2][1]) * mult;
    result.y = (m[2][0] - m[0][2]) * mult;
    result.z = (m[0][1] - m[1][0]) * mult;
  of 1:
    result.w = (m[1][2] - m[2][1]) * mult;
    result.x = biggestVal;
    result.y = (m[0][1] + m[1][0]) * mult;
    result.z = (m[2][0] + m[0][2]) * mult;
  of 2:
    result.w = (m[2][0] - m[0][2]) * mult;
    result.x = (m[0][1] + m[1][0]) * mult;
    result.y = biggestVal;
    result.z = (m[1][2] + m[2][1]) * mult;
  of 3:
    result.w = (m[0][1] - m[1][0]) * mult;
    result.x = (m[2][0] + m[0][2]) * mult;
    result.y = (m[1][2] + m[2][1]) * mult;
    result.z = biggestVal;

proc quat*[T](mat: Mat4[T]): Quat[T] =
  ## mat needs to be rotation matrix (orthogonal, det(mat) = 1
  quat(mat3(mat[0].xyz, mat[1].xyz, mat[2].xyz))


proc `*`*[T](p,q: Quat[T]): Quat[T] =
  result.w = p.w * q.w - p.x * q.x - p.y * q.y - p.z * q.z
  result.x = p.w * q.x + p.x * q.w + p.y * q.z - p.z * q.y
  result.y = p.w * q.y + p.y * q.w + p.z * q.x - p.x * q.z
  result.z = p.w * q.z + p.z * q.w + p.x * q.y - p.y * q.x

proc `*`*[T](q : Quat[T], s : T) : Quat[T] =
  for i in 0 .. 3:
    result[i] = q[i] * s

proc `*`*[T](s:T, q : Quat[T]) : Quat[T] =
  for i in 0 .. 3:
    result[i] = q[i] * s


proc `*`*[T](q: Quat[T], v: Vec3[T]): Vec3[T] =
  let QuatVector = vec3(q.x, q.y, q.z)
  let uv = cross(QuatVector, v)
  let uuv = cross(QuatVector, uv)

  return v + ((uv * q.w) + uuv) * T(2)

proc `*`*[T](v: Vec3[T]; q: Quat[T]): Vec3[T] =
  return inverse(q) * v

proc `*`*[T](q: Quat[T], v: Vec4[T]): Vec4[T] =
  vec4(q * v.xyz, v.w);

proc `*`*[T](v: Vec4[T]; q: Quat[T]): Vec4[T] =
  inverse(q) * v


proc `/`*[T](q : Quat[T]; s: T): Quat[T] =
  for i in 0 .. 3:
    result[i] = q[i] / s

proc `+`*[T](q1,q2 : Quat[T]) : Quat[T] =
  for i in 0 .. 3:
    result[i] = q1[i] + q2[i]

proc `-`*[T](q1,q2 : Quat[T]) : Quat[T] =
  for i in 0 .. 3:
    result[i] = q1[i] - q2[i]

proc `-`*[T](q : Quat[T]) : Quat[T] =
  for i in 0 .. 3:
    result[i] = -q[i]

proc `*=`*[T](q1: var Quat[T]; q2: Quat[T]): void =
  q1 = q1 * q2

proc `+=`*[T](q1: var Quat[T]; q2: Quat[T]): void =
  q1 = q1 + q2

proc `-=`*[T](q1: var Quat[T]; q2: Quat[T]): void =
  q1 = q1 - q2

proc length2*[T](q : Quat[T]) : T =
  for i in 0 .. 3:
    result += q[i] * q[i]

proc length*[T](q : Quat[T]) : T =
  q.length2.sqrt

proc normalize*[T](q : Quat[T]) : Quat[T] =
  q * (1.0f / q.length)

proc angle*[T](x: Quat[T]): T =
  return arccos(x.w) * T(2)

proc axis*[T](x: Quat[T]): Vec3[T] =
  let tmp1: T = T(1) - x.w * x.w;
  if tmp1 <= T(0):
    return vec3(T(0), T(0), T(1))
  let tmp2: T = T(1) / sqrt(tmp1)
  vec3(x.x * tmp2, x.y * tmp2, x.z * tmp2)

proc roll*[T](q: Quat[T]): T =
  T(arctan2(T(2) * (q.x * q.y + q.w * q.z), q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z))

proc pitch*[T](q: Quat[T]): T =
  let y: T = T(2) * (q.y * q.z + q.w * q.x);
  let x: T = q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z;
  if y == T(0) and x == T(0): # avoid atan2(0,0) - handle singularity - Matiis
    T(T(2)*arctan2(q.x,q.w))
  else:
    T(arctan2(y,x))

proc yaw*[T](q: Quat[T]): T =
  arcsin(clamp(T(-2) * (q.x * q.z - q.w * q.y), T(-1), T(1)))

proc eulerAngles*[T](x: Quat[T]): Vec3[T] =
  vec3(pitch(x), yaw(x), roll(x))

proc dot*[T](a, b: Quat[T]): T {.inline.} =
  a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w

const epsilon = 0.0001


proc fastMix*[T](a,b: Quat[T]; alpha: T) : Quat[T] =
  ## Returns a normalized linear interpolated quaternion of x and y according a.
  normalize(a * (1 - alpha) + b * alpha)

proc mix*[T](x,y: Quat[T]; a: T): Quat[T] =
  let cosTheta: T = dot(x, y)
  # Perform a linear interpolation when cosTheta is close to 1 to avoid side effect of sin(angle) becoming a zero denominator
  if cosTheta > T(1) - T(epsilon):
    # Linear interpolation
    return quat(
      mix(x.x, y.x, a),
      mix(x.y, y.y, a),
      mix(x.z, y.z, a),
      mix(x.w, y.w, a))
  else:
    # Essential Mathematics, page 467
    let angle: T = arccos(cosTheta);
    return (sin((T(1) - a) * angle) * x + sin(a * angle) * y) / sin(angle)


proc slerp*[T](x,y: Quat[T]; a: T): Quat[T] =
  ## same as mix, just that it ensures to take the sort path around the sphere
  var z = y
  var cosTheta: T = dot(x, y)

  if cosTheta < T(0):
    z = -y
    cosTheta = -cosTheta

  # Perform a linear interpolation when cosTheta is close to 1 to avoid side effect of sin(angle) becoming a zero denominator
  if cosTheta > T(1) - T(epsilon):
    # Linear interpolation
    return quat(
      mix(x.x, z.x, a),
      mix(x.y, z.y, a),
      mix(x.z, z.z, a),
      mix(x.w, z.w, a))
  else:
    # Essential Mathematics, page 467
    let angle: T = arccos(cosTheta);
    return (sin((T(1) - a) * angle) * x + sin(a * angle) * z) / sin(angle)

proc conjugate[T](q: Quat[T]): Quat[T] =
  result.arr[0] = -q[0]
  result.arr[1] = -q[1]
  result.arr[2] = -q[2]
  result.arr[3] =  q[3]

proc inverse*[T](q: Quat[T]) : Quat[T] =
  return conjugate(q) / dot(q, q);

proc rotate*[T](q: Quat[T]; angle: T; v: Vec3[T]) : Quat[T] =
  ## rotates q around the axis v by the given angle in radians
  ## normalizes ``v`` for you.
  result.arr = vec4(
    normalize(v) * sin(angle * 0.5),
    cos(angle * 0.5)
  ).arr
  result = q * result

proc mat3*[T](q : Quat[T]) : Mat3[T] =
  let
    txx = 2*q.x*q.x
    tyy = 2*q.y*q.y
    tzz = 2*q.z*q.z
    txy = 2*q.x*q.y
    txz = 2*q.x*q.z
    tyz = 2*q.y*q.z
    txw = 2*q.x*q.w
    tyw = 2*q.y*q.w
    tzw = 2*q.z*q.w

  result[0] = vec3(1 - tyy - tzz,     txy + tzw,     txz - tyw);
  result[1] = vec3(    txy - tzw, 1 - txx - tzz,     tyz + txw);
  result[2] = vec3(    txz + tyw,     tyz - txw, 1 - txx - tyy);

proc mat4*[T](q: Quat[T]; v: Vec4[T]): Mat4[T] =
  let tmp = q.mat3
  result[0] = vec4(tmp[0],0)
  result[1] = vec4(tmp[1],0)
  result[2] = vec4(tmp[2],0)
  result[3] = v

proc mat4*[T](q: Quat[T]; v: Vec3[T]): Mat4[T] = mat4(q, vec4(v,1))

proc mat4*[T](q: Quat[T]): Mat4[T] = mat4(q, vec4[T](0,0,0,1))

proc poseMatrix*[T](translate: Vec3[T]; rotate: Quat[T]; scale: Vec3[T]): Mat4[T] =
  let
    factor1 = rotate.normalize.mat3
    factor2 = scale.diag

  let scalerot_mat = factor1 * factor2

  result[0] = vec4(scalerot_mat[0], 0)
  result[1] = vec4(scalerot_mat[1], 0)
  result[2] = vec4(scalerot_mat[2], 0)
  result[3] = vec4(translate,    1)

proc quat*[T](u,v: Vec3[T]): Quat[T] =
  let LocalW: Vec3[T] = cross(u,v)
  let Dot: T = dot(u,v)
  let q = quat(LocalW.x, LocalW.y, LocalW.z, T(1) + Dot)
  normalize(q)

type
  Quatf* = Quat[float32]
  Quatd* = Quat[float64]

proc quatf*(x,y,z,w : float32) : Quatf {.inline.} =  Quatf(arr: [x,y,z,w])
proc quatf*(axis: Vec3f; angle: float32): Quatf = quat[float32](axis,angle)
proc quatf*(u,v: Vec3f): Quatf = quat(u,v)
proc quatf*(mat: Mat3f): Quatf = quat[float32](mat)
proc quatf*(): Quatf = Quatf(arr: [0.0f,0,0,1])

proc quatd*(x,y,z,w : float64) : Quatd {.inline.} =  Quatd(arr: [x,y,z,w])
proc quatd*(axis: Vec3d; angle: float64): Quatd = quat[float64](axis,angle)
proc quatd*(u,v: Vec3d): Quatd = quat(u,v)
proc quatd*(mat: Mat3d): Quatd = quat[float64](mat)
proc quatd*(): Quatd = Quatd(arr: [0.0d,0,0,1])

#[
proc frustum*[T](left, right, bottom, top, near, far: T): Mat4[T] =
  result[0][0] =       (2*near)/(right-left)
  result[1][1] =       (2*near)/(top-bottom)
  result[2][2] =     (far+near)/(near-far)
  result[2][0] =   (right+left)/(right-left)
  result[2][1] =   (top+bottom)/(top-bottom)
  result[2][3] = -1
  result[3][2] =   (2*far*near)/(near-far)
]#

when isMainModule:
  let q1 = quatf(1,2,3,4)
  let q2 = inverse(q1)

  echo q1 * q2
  echo q2 * q1

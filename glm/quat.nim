#quaternion

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
    
    
proc `$`*[T](q : Quat[T]) : string = q.arr.mkString("quatf(", ", ", ")")

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
  result.arr = [x,y,z,w]
  
proc quat*[T](axis: Vec3[T]; angle: T): Quat[T] =
  let s = sin(angle / 2)
  result.x = axis.x * s
  result.y = axis.y * s
  result.z = axis.z * s
  result.w = cos(angle / 2)
  
proc quat*[T](mat: Mat3[T]): Quat[T] =
  ## mat needs to be rotation matrix (orthogonal, det(mat) = 1
  let qw = sqrt(1 + mat[0][0] + mat[1][1] + mat[2][2]) * 0.5'f32
  result.x = (mat[1][2] - mat[2][1]) / (4 * qw)
  result.y = (mat[2][0] - mat[0][2]) / (4 * qw)
  result.z = (mat[0][1] - mat[1][0]) / (4 * qw)
  result.w = qw
  
# untestet
proc `*`*[T](q1,q2 : Quat[T]) : Quat[T] = 
  result.arr[0] = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y
  result.arr[1] = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x
  result.arr[2] = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w
  result.arr[3] = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z


proc `*`*[T](q : Quat[T], s : T) : Quat[T] =
  for i in 0 .. 3:
    result[i] = q[i] * s

proc `+`*[T](q1,q2 : Quat[T]) : Quat[T] =
  for i in 0 .. 3:
    result[i] = q1[i] + q2[i]

proc `-`*[T](q1,q2 : Quat[T]) : Quat[T] =
  for i in 0 .. 3:
    result[i] = q1[i] - q2[1]

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

proc mix*[S,T](a,b: S; alpha: T) : S =
  a * (1 - alpha) + b * alpha

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

proc mat4*[T](q: Quat[T]; v: Vec4[T] = vec4(T(0),0,0,1)): Mat4f =
  let tmp = q.mat3
  result[0] = vec4(tmp[0],0)
  result[1] = vec4(tmp[1],0)
  result[2] = vec4(tmp[2],0)
  result[3] = v

proc mat4*[T](q: Quat[T]; v: Vec3[T]): Mat4[T] = mat4(q, vec4(v,1))
  
proc poseMatrix*[T](translate: Vec3[T]; rotate: Quat[T]; scale: Vec3[T]): Mat4[T] =
  let
    factor1 = rotate.normalize.mat3
    factor2 = scale.diag
  
  let scalerot_mat = factor1 * factor2
  
  result[0] = vec4(scalerot_mat[0], 0)
  result[1] = vec4(scalerot_mat[1], 0)
  result[2] = vec4(scalerot_mat[2], 0)
  result[3] = vec4(translate,    1)


type
  Quatf* = Quat[float32]
  Quatd* = Quat[float64]

proc quatf*(x,y,z,w : float32) : Quatf {.inline.} =  Quatf(arr: [x,y,z,w])
proc quatf*(axis: Vec3f; angle: float32): Quatf = quat[float32](axis,angle)
proc quatf*(mat: Mat3f): Quatf = quat[float32](mat)

proc quatd*(x,y,z,w : float64) : Quatd {.inline.} =  Quatd(arr: [x,y,z,w])
proc quatd*(axis: Vec3d; angle: float64): Quatd = quat[float64](axis,angle)
proc quatd*(mat: Mat3d): Quatd = quat[float64](mat)

  
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

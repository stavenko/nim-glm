import strutils, macros
#import ./arrayUtils
import macros.matrix
import vec

type
  Mat[M,N: static[int]; T] = object
    arr: array[M, Vec[N,T]]

proc `$`(m: Mat): string =
  var cols: array[m.M, array[m.N, string]]
  for i, col in m.arr:
    cols[i] = columnFormat(col)

  result = ""
  for row in 0 ..< m.N:
    result &= '['
    for col in 0 ..< m.M:
      if col != 0:
        result &= ", "
      result &= cols[col][row]
    result &= "]\n"         
    
proc `[]=`*[M,N,T](v:var Mat[M,N,T]; ix:int; c:Vec[N,T]): void {.inline.} =
    v.arr[ix] = c
proc `[]`*[M,N,T](v: Mat[M,N,T]; ix: int): Vec[N,T] {.inline.} =
  v.arr[ix]
proc `[]`*[M,N,T](v: var Mat[M,N,T]; ix: int): var Vec[N,T] {.inline.} =
  v.arr[ix]

proc `[]=`*[M,N,T](v:var Mat[M,N,T]; ix, iy:int; value:T): void {.inline.} =
    v.arr[ix].arr[iy] = value
proc `[]`*[M,N,T](v: Mat[M,N,T]; ix, iy: int): T {.inline.} =
  v.arr[ix].arr[iy]
proc `[]`*[M,N,T](v: var Mat[M,N,T]; ix, iy: int): var T {.inline.} =
  v.arr[ix].arr[iy]

  
proc caddr[M,N,T](m: var Mat[M,N,T]): ptr T = m.arr[0].arr[0].addr

##############
# type alias #
##############

type
  Mat4*[T] = Mat[4,4,T]
  Mat3*[T] = Mat[3,3,T]
  Mat2*[T] = Mat[2,2,T]
  
type
  Mat4f*  = Mat[4, 4, float32]
  Mat3f*  = Mat[3, 3, float32]
  Mat2f*  = Mat[2, 2, float32]
  Mat4d*  = Mat[4, 4, float64]
  Mat3d*  = Mat[3, 3, float64]
  Mat2d*  = Mat[2, 2, float64]
  Mat4i*  = Mat[4, 4, int32]
  Mat3i*  = Mat[3, 3, int32]
  Mat2i*  = Mat[2, 2, int32]
  Mat4l*  = Mat[4, 4, int64]
  Mat3l*  = Mat[3, 3, int64]
  Mat2l*  = Mat[2, 2, int64]

proc diag*[N,M,T](m : Mat[N,M,T]): Vec[min(N,M), T] =
  for i in 0 ..< min(N,M):
    result.arr[i] = m.arr[i].arr[i]

proc diag*[N,T](v : Vec[N,T]): Mat[N,N,T] =
  for i in 0 ..< N:
    result.arr[i].arr[i] = v.arr[i]

####################
# type constructor #
####################  
  
proc mat4[T](a,b,c,d: Vec4[T]) : Mat4[T] =
  result.arr = [a,b,c,d]
  
proc mat3[T](a,b,c: Vec3[T]) : Mat3[T] =
  result.arr = [a,b,c]

proc mat2[T](a,b,c,d: Vec2[T]) : Mat2[T] =
  result.arr = [a,b]

proc mat4[T](v: Vec4[T]): Mat4[T] =
  for i in 0 .. 3:
    result.arr[i].arr[i] = v.arr[i]

proc mat3[T](v: Vec3[T]): Mat3[T] =
  for i in 0 .. 2:
    result.arr[i].arr[i] = v.arr[i]
  
proc mat2[T](v: Vec2[T]): Mat2[T] =
  for i in 0 .. 1:
    result.arr[i].arr[i] = v.arr[i]
  
proc mat4[T](s: T): Mat4[T] =
  for i in 0 .. 3:
    result.arr[i].arr[i] = s

proc mat3[T](s: T): Mat3[T] =
  for i in 0 .. 2:
    result.arr[i].arr[i] = s

proc mat2[T](s: T): Mat2[T] =
  for i in 0 .. 1:
    result.arr[i].arr[i] = s

  
proc mat4f(a,b,c,d: Vec4f) : Mat4f =
  result.arr = [a,b,c,d]

proc mat3f(a,b,c: Vec3f) : Mat3f =
  result.arr = [a,b,c]

proc mat2f(a,b: Vec2f) : Mat2f =
  result.arr = [a,b]

proc mat4d(a,b,c,d: Vec4d) : Mat4d =
  result.arr = [a,b,c,d]

proc mat3d(a,b,c: Vec3d) : Mat3d =
  result.arr = [a,b,c]

proc mat2d(a,b: Vec2d) : Mat2d =
  result.arr = [a,b]

proc mat4i(a,b,c,d: Vec4i) : Mat4i =
  result.arr = [a,b,c,d]

proc mat3i(a,b,c: Vec3i) : Mat3i =
  result.arr = [a,b,c]

proc mat2i(a,b: Vec2i) : Mat2i =
  result.arr = [a,b]

proc mat4l(a,b,c,d: Vec4l) : Mat4l =
  result.arr = [a,b,c,d]

proc mat3l(a,b,c: Vec3l) : Mat3l =
  result.arr = [a,b,c]

proc mat2l(a,b: Vec2l) : Mat2l =
  result.arr = [a,b]

#emptyConstructors(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE)
#diagonalConstructors(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE)
#matrixComparison(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE)

proc det[T](m: Mat2[T]): T =
   m[0][0] * m[1][1] - m[1][0] * m[0][1]

proc det[T](m: Mat3[T]): T = (
  + m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
    m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2]) +
    m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2])
)

proc det[T](m: Mat4[T]): T =
  var tmpOuter : Mat2[T]

  for i in 0 .. 1:
    for j in 0 .. 1:
      var  tmpInner : Mat2[T]

      for k in 0 .. 1:
        for l in 0 .. 1:
          tmpInner[k][l] = m[2*i + k][2*j + l]

      tmpOuter[i,j] = det(tmpInner)

  result = det(tmpOuter)

proc inverse*[T](m: Mat2[T]): Mat2[T]=
  # one over determinat
  
  let od = T(1) / det(m)

  result[0,0] =   m[1,1] * od
  result[0,1] = - m[0,1] * od
  result[1,0] = - m[1,0] * od
  result[1,1] =   m[0,0] * od

proc inverse*[T](m: Mat3[T]): Mat3[T]=
  # one over determinant
  let od = (1.T) / det(m)
              
  result[0,0] = + (m[1,1] * m[2,2] - m[2,1] * m[1,2]) * oD
  result[1,0] = - (m[1,0] * m[2,2] - m[2,0] * m[1,2]) * oD
  result[2,0] = + (m[1,0] * m[2,1] - m[2,0] * m[1,1]) * oD
  result[0,1] = - (m[0,1] * m[2,2] - m[2,1] * m[0,2]) * oD
  result[1,1] = + (m[0,0] * m[2,2] - m[2,0] * m[0,2]) * oD
  result[2,1] = - (m[0,0] * m[2,1] - m[2,0] * m[0,1]) * oD
  result[0,2] = + (m[0,1] * m[1,2] - m[1,1] * m[0,2]) * oD
  result[1,2] = - (m[0,0] * m[1,2] - m[1,0] * m[0,2]) * oD
  result[2,2] = + (m[0,0] * m[1,1] - m[1,0] * m[0,1]) * oD

proc inverse*[T](m: Mat4[T]):Mat4[T]=
  let
    Coef00:T = (m[2,2]  * m[3,3]) - (m[3,2]  *  m[2,3])
    Coef02:T = (m[1,2]  * m[3,3]) - (m[3,2]  *  m[1,3])
    Coef03:T = (m[1,2]  * m[2,3]) - (m[2,2]  *  m[1,3])

    Coef04:T = (m[2,1]  * m[3,3]) - (m[3,1]  *  m[2,3])
    Coef06:T = (m[1,1]  * m[3,3]) - (m[3,1]  *  m[1,3])
    Coef07:T = (m[1,1]  * m[2,3]) - (m[2,1]  *  m[1,3])

    Coef08:T = (m[2,1]  * m[3,2]) - (m[3,1]  *  m[2,2])
    Coef10:T = (m[1,1]  * m[3,2]) - (m[3,1]  *  m[1,2])
    Coef11:T = (m[1,1]  * m[2,2]) - (m[2,1]  *  m[1,2])

    Coef12:T = (m[2,0]  * m[3,3]) - (m[3,0]  *  m[2,3])
    Coef14:T = (m[1,0]  * m[3,3]) - (m[3,0]  *  m[1,3])
    Coef15:T = (m[1,0]  * m[2,3]) - (m[2,0]  *  m[1,3])

    Coef16:T = (m[2,0]  * m[3,2]) - (m[3,0]  *  m[2,2])
    Coef18:T = (m[1,0]  * m[3,2]) - (m[3,0]  *  m[1,2])
    Coef19:T = (m[1,0]  * m[2,2]) - (m[2,0]  *  m[1,2])

    Coef20:T = (m[2,0]  * m[3,1]) - (m[3,0]  *  m[2,1])
    Coef22:T = (m[1,0]  * m[3,1]) - (m[3,0]  *  m[1,1])
    Coef23:T = (m[1,0]  * m[2,1]) - (m[2,0]  *  m[1,1])

  var
    Fac0 = vec4(Coef00, Coef00, Coef02, Coef03)
    Fac1 = vec4(Coef04, Coef04, Coef06, Coef07)
    Fac2 = vec4(Coef08, Coef08, Coef10, Coef11)
    Fac3 = vec4(Coef12, Coef12, Coef14, Coef15)
    Fac4 = vec4(Coef16, Coef16, Coef18, Coef19)
    Fac5 = vec4(Coef20, Coef20, Coef22, Coef23)

    Vec0=vec4(m[1,0], m[0,0], m[0,0], m[0,0])
    Vec1=vec4(m[1,1], m[0,1], m[0,1], m[0,1])
    Vec2=vec4(m[1,2], m[0,2], m[0,2], m[0,2])
    Vec3=vec4(m[1,3], m[0,3], m[0,3], m[0,3])

    Inv0=vec4((Vec1 * Fac0) - (Vec2 * Fac1) + (Vec3 * Fac2))
    Inv1=vec4((Vec0 * Fac0) - (Vec2 * Fac3) + (Vec3 * Fac4))
    Inv2=vec4((Vec0 * Fac1) - (Vec1 * Fac3) + (Vec3 * Fac5))
    Inv3=vec4((Vec0 * Fac2) - (Vec1 * Fac4) + (Vec2 * Fac5))

    SignA:Vec4[T] = vec4(+1.T, -1, +1, -1)
    SignB:Vec4[T] = vec4(-1.T, +1, -1, +1)
    Inverse = mat4(Inv0 * SignA, Inv1 * SignB, Inv2 * SignA, Inv3 * SignB)

    Row0 = vec4(Inverse[0,0], Inverse[1,0], Inverse[2,0], Inverse[3,0])

    Dot0 = m[0] * Row0
    Dot1 = (Dot0.x + Dot0.y) + (Dot0.z + Dot0.w)

    OneOverDeterminant = (1.T) / Dot1
  result = Inverse * OneOverDeterminant

#fromArray(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE)

proc getRow*[M,N,T](m: Mat[M,N,T]; i: int): Vec[M,T] =
  for j in 0 ..< M:
    result.arr[j] = m.arr[j].arr[i]
  
proc transpose*[M,N,T](m: Mat[M,N,T]): Mat[N,M,T] =
  for i in 0 ..< N:
    result.arr[i] = m.getRow(i)
  
proc `*`*[M,N,T](m: Mat[M,N,T]; v: Vec[M, T]): Vec[N, T] =
  for i in 0 ..< M:
    result += m.arr[i] * v.arr[i]

proc `*`*[M,N,T](v: Vec[N,T]; m: Mat[M,N,T]): Vec[M, T] =
  for i in 0 ..< M:
    result.arr[i] = dot(v, m.arr[i])

proc `*`*[M,N,O,T](m1: Mat[M,N,T]; m2: Mat[O,M,T]): Mat[N,O,T] =
  for i in 0 ..< N:
    result.arr[i] = m1.getRow(i) * m2

proc `*`*[M,N,T](m: Mat[M,N,T]; s: T): Mat[M,N,T] =
  for i in 0 ..< M:
    result.arr[i] = m.arr[i] * s

proc `*`*[M,N,T](s: T; m: Mat[M,N,T]): Mat[M,N,T] =
  for i in 0 ..< M:
    result.arr[i] = s * m.arr[i]

proc `+`*[M,N,T](m1,m2: Mat[M,N,T]): Mat[M,N,T] =
  for i in 0 ..< M:
    result.arr[i] = m1.arr[i] + m2.arr[i]

proc `-`*[M,N,T](m1,m2: Mat[M,N,T]): Mat[M,N,T] =
  for i in 0 ..< M:
    result.arr[i] = m1.arr[i] - m2.arr[i]

proc `*=`*[M,N,T](m: var Mat[M,N,T]; s: T): void =
  for i in 0 ..< M:
    m.arr[i] *= s

proc `*=`*[N,T](m1: var Mat[N,N,T]; m2: Mat[N,N,T]): void =
  for i in 0 ..< N:
    m1.arr
    
    
    
if isMainModule:

  var mats : array[2, Mat4f]  

  for m in mats.mitems:
    var x = 0
    for i in 0 .. 3:
      for j in 0 .. 3:
        m[i,j] = float32(x)
        x += 1

  echo mats[0]
  echo mats[1]

  echo mats[0] * mats[1]

  let m22 = mat3(vec3(1.0, 5, 10), vec3(0.66,1,70), vec3(10.0,2.0,1))
  let m22i = inverse(m22)
  let v2  = vec3(2.0)

  echo m22 * m22i
    
  let v2m = v2 * m22
  let v2r = v2m * m22i

  var m22v = m22
  m22v *= 3
  echo m22v

  echo mat3(5.0)

  echo det(diag(vec4(1,2,3,4)))

#type
#  SomeVec* = Vec4 | Vec3 | Vec2
#  SomeMat* = Mat4x4 | Mat4x3 | Mat4x2 | Mat3x4 | Mat3x3 | Mat3x2 | Mat2x4 | Mat2x3 | Mat2x2

template numCols*[N,M,T](t : typedesc[Mat[N,M,T]]): int = N
template numRows*[N,M,T](t : typedesc[Mat[N,M,T]]): int = M

    

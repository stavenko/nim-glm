when not compiles(SomeFloat):
  type SomeFloat = SomeReal

import vec

type
  Mat*[M,N: static[int]; T] = object
    arr*: array[M, Vec[N,T]]

when defined(noUnicode) or defined(windows):
  # On a windows terminal, we are still in the 80s
  const matrixDecoration = [" / ", " \\ ", "|  ", " \\ ", " / ", "  |"]
else:
  const matrixDecoration = ["⎡", "⎣", "⎢", "⎤", "⎦", "⎥"]

proc `$`*(m: Mat): string =
  var cols: array[m.M, array[m.N, string]]
  for i, col in m.arr:
    cols[i] = columnFormat(col)

  result = ""
  for row in 0 ..< m.N:
    if row == 0:
      result &= matrixDecoration[0]
    elif row == m.N - 1:
      result &= matrixDecoration[1]
    else:
      result &= matrixDecoration[2]

    for col in 0 ..< m.M:
      if col != 0:
        result &= "  "
      result &= cols[col][row]

    if row == 0:
      result &= matrixDecoration[3]
    elif row == m.N - 1:
      result &= matrixDecoration[4]
    else:
      result &= matrixDecoration[5]

    result &= "\n"

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

proc caddr*[M,N,T](m: var Mat[M,N,T]): ptr T = m.arr[0].arr[0].addr

##############
# type alias #
##############

type
  Mat4x4*[T] = Mat[4,4,T]
  Mat3x4*[T] = Mat[3,4,T]
  Mat2x4*[T] = Mat[2,4,T]
  Mat4x3*[T] = Mat[4,3,T]
  Mat3x3*[T] = Mat[3,3,T]
  Mat2x3*[T] = Mat[2,3,T]
  Mat4x2*[T] = Mat[4,2,T]
  Mat3x2*[T] = Mat[3,2,T]
  Mat2x2*[T] = Mat[2,2,T]

  Mat4*[T] = Mat[4,4,T]
  Mat3*[T] = Mat[3,3,T]
  Mat2*[T] = Mat[2,2,T]

proc diag*[M,N,T](m : Mat[M,N,T]): Vec[min(M,N), T] =
  for i in 0 ..< min(M,N):
    result.arr[i] = m.arr[i].arr[i]

proc `diag=`*[M,N,T,U](m : var Mat[M,N,T], v: Vec[U, T]) =
  static:
    assert U == min(M,N)
  for i in 0 ..< U:
    m.arr[i].arr[i] = v.arr[i]

proc diag*[N,T](v : Vec[N,T]): Mat[N,N,T] =
  for i in 0 ..< N:
    result.arr[i].arr[i] = v.arr[i]

####################
# type constructor #
####################

# generic
proc mat4*[T](a,b,c,d: Vec4[T]) : Mat4[T] =
  result.arr = [a,b,c,d]

proc mat3*[T](a,b,c: Vec3[T]) : Mat3[T] =
  result.arr = [a,b,c]

proc mat2*[T](a,b: Vec2[T]) : Mat2[T] =
  result.arr = [a,b]


proc mat4x4*[T](a,b,c,d: Vec4[T]) : Mat4x4[T] =
  result.arr = [a,b,c,d]

proc mat4x3*[T](a,b,c,d: Vec3[T]) : Mat4x3[T] =
  result.arr = [a,b,c,d]

proc mat4x2*[T](a,b,c,d: Vec2[T]) : Mat4x2[T] =
  result.arr = [a,b,c,d]


proc mat3x4*[T](a,b,c: Vec4[T]) : Mat3x4[T] =
  result.arr = [a,b,c]

proc mat3x3*[T](a,b,c: Vec3[T]) : Mat3x3[T] =
  result.arr = [a,b,c]

proc mat3x2*[T](a,b,c: Vec2[T]) : Mat3x2[T] =
  result.arr = [a,b,c]


proc mat2x4*[T](a,b: Vec4[T]) : Mat2x4[T] =
  result.arr = [a,b]

proc mat2x3*[T](a,b: Vec3[T]) : Mat2x3[T] =
  result.arr = [a,b]

proc mat2x2*[T](a,b: Vec2[T]) : Mat2x2[T] =
  result.arr = [a,b]


proc mat4*[T](v: Vec4[T]): Mat4[T] =
  for i in 0 .. 3:
    result.arr[i].arr[i] = v.arr[i]

proc mat3*[T](v: Vec3[T]): Mat3[T] =
  for i in 0 .. 2:
    result.arr[i].arr[i] = v.arr[i]

proc mat2*[T](v: Vec2[T]): Mat2[T] =
  for i in 0 .. 1:
    result.arr[i].arr[i] = v.arr[i]


proc mat4*[T](s: T): Mat4[T] =
  for i in 0 .. 3:
    result.arr[i].arr[i] = s

proc mat3*[T](s: T): Mat3[T] =
  for i in 0 .. 2:
    result.arr[i].arr[i] = s

proc mat2*[T](s: T): Mat2[T] =
  for i in 0 .. 1:
    result.arr[i].arr[i] = s

proc mat4*[T]() : Mat4[T] =
  for i in 0 .. 3:
    result.arr[i].arr[i] = T(1)

proc mat3*[T]() : Mat3[T] =
  for i in 0 .. 2:
    result.arr[i].arr[i] = T(1)

proc mat2*[T]() : Mat2[T] =
  for i in 0 .. 1:
    result.arr[i].arr[i] = T(1)
template genMats(suffix:untyped,valtype:typed):untyped=
  type
    `Mat4 suffix`*   {.inject.} = Mat[4, 4, valtype]
    `Mat3 suffix`*   {.inject.} = Mat[3, 3, valtype]
    `Mat2 suffix`*   {.inject.} = Mat[2, 2, valtype]
    `Mat4x4 suffix`* {.inject.} = Mat[4, 4, valtype]
    `Mat3x4 suffix`* {.inject.} = Mat[3, 4, valtype]
    `Mat2x4 suffix`* {.inject.} = Mat[2, 4, valtype]
    `Mat4x3 suffix`* {.inject.} = Mat[4, 3, valtype]
    `Mat3x3 suffix`* {.inject.} = Mat[3, 3, valtype]
    `Mat2x3 suffix`* {.inject.} = Mat[2, 3, valtype]
    `Mat4x2 suffix`* {.inject.} = Mat[4, 2, valtype]
    `Mat3x2 suffix`* {.inject.} = Mat[3, 2, valtype]
    `Mat2x2 suffix`* {.inject.} = Mat[2, 2, valtype]
  template `mat2 suffix`*(                )  : untyped = mat2[valtype](        )
  template `mat3 suffix`*(                )  : untyped = mat3[valtype](        )
  template `mat4 suffix`*(                )  : untyped = mat4[valtype](        )
  template `mat2 suffix`*(a:       untyped)  : untyped = mat2[valtype](a       )
  template `mat3 suffix`*(a:       untyped)  : untyped = mat3[valtype](a       )
  template `mat4 suffix`*(a:       untyped)  : untyped = mat4[valtype](a       )
  template `mat2 suffix`*(a,b:     untyped)  : untyped = mat2[valtype](a,b     )
  template `mat3 suffix`*(a,b,c:   untyped)  : untyped = mat3[valtype](a,b,c   )
  template `mat4 suffix`*(a,b,c,d: untyped)  : untyped = mat4[valtype](a,b,c,d )

genMats f, float32
genMats d, float64
genMats i, int32
genMats l, int64
genMats ui, uint32
genMats ul, uint64

proc det*[T](m: Mat2[T]): T =
   m[0,0] * m[1,1] - m[1,0] * m[0,1]

proc det*[T](m: Mat3[T]): T =
  + m[0,0] * (m[1,1] * m[2,2] - m[2,1] * m[1,2]) -
    m[1,0] * (m[0,1] * m[2,2] - m[2,1] * m[0,2]) +
    m[2,0] * (m[0,1] * m[1,2] - m[1,1] * m[0,2])

proc det*[T](m: Mat4[T]): T =
  if m[0,0] != 0:
    result += m[0,0] * det(mat3(m[1].yzw, m[2].yzw, m[3].yzw))
  if m[0,1] != 0:
    result -= m[0,1] * det(mat3(m[1].xzw, m[2].xzw, m[3].xzw))
  if m[0,2] != 0:
    result += m[0,2] * det(mat3(m[1].xyw, m[2].xyw, m[3].xyw))
  if m[0,3] != 0:
    result -= m[0,3] * det(mat3(m[1].xyz, m[2].xyz, m[3].xyz))

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
    Coef00:T = (m[2,2] * m[3,3]) - (m[3,2] * m[2,3])
    Coef02:T = (m[1,2] * m[3,3]) - (m[3,2] * m[1,3])
    Coef03:T = (m[1,2] * m[2,3]) - (m[2,2] * m[1,3])

    Coef04:T = (m[2,1] * m[3,3]) - (m[3,1] * m[2,3])
    Coef06:T = (m[1,1] * m[3,3]) - (m[3,1] * m[1,3])
    Coef07:T = (m[1,1] * m[2,3]) - (m[2,1] * m[1,3])

    Coef08:T = (m[2,1] * m[3,2]) - (m[3,1] * m[2,2])
    Coef10:T = (m[1,1] * m[3,2]) - (m[3,1] * m[1,2])
    Coef11:T = (m[1,1] * m[2,2]) - (m[2,1] * m[1,2])

    Coef12:T = (m[2,0] * m[3,3]) - (m[3,0] * m[2,3])
    Coef14:T = (m[1,0] * m[3,3]) - (m[3,0] * m[1,3])
    Coef15:T = (m[1,0] * m[2,3]) - (m[2,0] * m[1,3])

    Coef16:T = (m[2,0] * m[3,2]) - (m[3,0] * m[2,2])
    Coef18:T = (m[1,0] * m[3,2]) - (m[3,0] * m[1,2])
    Coef19:T = (m[1,0] * m[2,2]) - (m[2,0] * m[1,2])

    Coef20:T = (m[2,0] * m[3,1]) - (m[3,0] * m[2,1])
    Coef22:T = (m[1,0] * m[3,1]) - (m[3,0] * m[1,1])
    Coef23:T = (m[1,0] * m[2,1]) - (m[2,0] * m[1,1])

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

    Inv0: Vec4[T] = (Vec1 * Fac0) - (Vec2 * Fac1) + (Vec3 * Fac2)
    Inv1: Vec4[T] = (Vec0 * Fac0) - (Vec2 * Fac3) + (Vec3 * Fac4)
    Inv2: Vec4[T] = (Vec0 * Fac1) - (Vec1 * Fac3) + (Vec3 * Fac5)
    Inv3: Vec4[T] = (Vec0 * Fac2) - (Vec1 * Fac4) + (Vec2 * Fac5)

    SignA: Vec4[T] = vec4(+1.T, -1, +1, -1)
    SignB: Vec4[T] = vec4(-1.T, +1, -1, +1)

    col0 : Vec4[T] = Inv0 * SignA
    col1 : Vec4[T] = Inv1 * SignB
    col2 : Vec4[T] = Inv2 * SignA
    col3 : Vec4[T] = Inv3 * SignB

    Inverse : Mat4[T] = mat4[T](col0, col1, col2, col3)

    Row0 = vec4(Inverse[0,0], Inverse[1,0], Inverse[2,0], Inverse[3,0])

    Dot0 = m[0] * Row0
    Dot1 = (Dot0.x + Dot0.y) + (Dot0.z + Dot0.w)

    OneOverDeterminant = (1.T) / Dot1
  result = Inverse * OneOverDeterminant

#fromArray(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE)

proc row*[M,N,T](m: Mat[M,N,T]; i: int): Vec[M,T] =
  for j in 0 ..< M:
    result.arr[j] = m.arr[j].arr[i]

proc `row0=`*[M,N,T](m: var Mat[M,N,T]; value: Vec[M,T]): void =
  for j in 0 ..< M:
    m.arr[j].arr[0] = value.arr[j]

proc `row1=`*[M,N,T](m: var Mat[M,N,T]; value: Vec[M,T]): void =
  for j in 0 ..< M:
    m.arr[j].arr[1] = value.arr[j]

proc `row2=`*[M,N,T](m: var Mat[M,N,T]; value: Vec[M,T]): void =
  for j in 0 ..< M:
    m.arr[j].arr[2] = value.arr[j]

proc `row3=`*[M,N,T](m: var Mat[M,N,T]; value: Vec[M,T]): void =
  for j in 0 ..< M:
    m.arr[j].arr[3] = value.arr[j]

proc transpose*[M,N,T](m: Mat[M,N,T]): Mat[N,M,T] =
  for i in 0 ..< N:
    result.arr[i] = m.row(i)

proc `*`*[M,N,T](m: Mat[M,N,T]; v: Vec[M, T]): Vec[N, T] =
  for i in 0 ..< M:
    result += m.arr[i] * v.arr[i]

proc `*`*[M,N,T](v: Vec[N,T]; m: Mat[M,N,T]): Vec[M, T] =
  for i in 0 ..< M:
    result.arr[i] = dot(v, m.arr[i])

proc `*`*[M,N,O,T](m1: Mat[M,N,T]; m2: Mat[O,M,T]): Mat[O,N,T] =
  for i in 0 ..< O:
    result.arr[i] = m1 * m2.arr[i]

proc `*`*[M,N,T](m: Mat[M,N,T]; s: T): Mat[M,N,T] =
  for i in 0 ..< M:
    result.arr[i] = m.arr[i] * s

proc `*`*[M,N,T](s: T; m: Mat[M,N,T]): Mat[M,N,T] =
  for i in 0 ..< M:
    result.arr[i] = s * m.arr[i]

proc `*=`*[M,N,T](m: var Mat[M,N,T]; s: T): void =
  for i in 0 ..< M:
    m.arr[i] *= s

proc `*=`*[N,T](m1: var Mat[N,N,T]; m2: Mat[N,N,T]): void =
  var tmp = m1 * m2;
  m1 = tmp

template foreachZipImpl(name,op: untyped): untyped =
  proc name*[M,N,T](m1,m2: Mat[M,N,T]): Mat[M,N,T] =
    for i in 0 ..< M:
      result.arr[i] = op(m1.arr[i], m2.arr[i])

foreachZipImpl(`+`,`+`)
foreachZipImpl(`-`,`-`)
foreachZipImpl(`.+`,`+`)
foreachZipImpl(`.-`,`-`)
foreachZipImpl(`.*`,`*`)
foreachZipImpl(`./`,`/`)
foreachZipImpl(matrixCompMult,`*`)

# conversions

proc mat4f*(mat: Mat4d): Mat4f {.inline.} = Mat4f(arr: [mat.arr[0].vec4f, mat.arr[1].vec4f, mat.arr[2].vec4f, mat.arr[3].vec4f])
proc mat4d*(mat: Mat4f): Mat4d {.inline.} = Mat4d(arr: [mat.arr[0].vec4d, mat.arr[1].vec4d, mat.arr[2].vec4d, mat.arr[3].vec4d])

template numCols*[M,N,T](t : typedesc[Mat[M,N,T]]): int = M
template numRows*[M,N,T](t : typedesc[Mat[M,N,T]]): int = N

when isMainModule:

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

  echo m22 * m22i

  let v2m = vec3(2.0) * m22
  let v2r = v2m * m22i

  var m22v = m22
  m22v *= 3
  echo m22v

  echo mat3(5.0)

  echo det(diag(vec4(1,2,3,4)))

  let v0 = vec4(3,0,5,6)
  let v1 = vec4(7,2,4,6)
  let v2 = vec4(3,-1,3,4)
  let v3 = vec4(0,1,2,-1)

  let someMat = mat4(v0,v1,v2,v3)

  echo "someMat:\n", someMat
  echo "det: ", someMat.det

  var mm : Mat3x2[int]
  mm[1] = vec2(33)

  echo mm

proc mix*[M,N,T](v1,v2: Mat[M,N,T]; a: T): Mat[M,N,T] =
  v1 * (1 - a) + v2 * a

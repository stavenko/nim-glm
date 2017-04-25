import vec

type
  Mat*[M,N: static[int]; T] = object
    arr*: array[M, Vec[N,T]]

proc `$`*(m: Mat): string =
  var cols: array[m.M, array[m.N, string]]
  for i, col in m.arr:
    cols[i] = columnFormat(col)

  result = ""
  for row in 0 ..< m.N:
    if row == 0:
      result &= "⎡"
    elif row == m.N - 1:
      result &= "⎣"
    else:
      result &= "⎢"

    for col in 0 ..< m.M:
      if col != 0:
        result &= "  "
      result &= cols[col][row]

    if row == 0:
      result &= "⎤\n"
    elif row == m.N - 1:
      result &= "⎦\n"
    else:
      result &= "⎥\n"

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


type
  # float32
  Mat4f*   = Mat[4,4,float32]
  Mat3f*   = Mat[3,3,float32]
  Mat2f*   = Mat[2,2,float32]
  Mat4x4f* = Mat[4,4,float32]
  Mat3x4f* = Mat[3,4,float32]
  Mat2x4f* = Mat[2,4,float32]
  Mat4x3f* = Mat[4,3,float32]
  Mat3x3f* = Mat[3,3,float32]
  Mat2x3f* = Mat[2,3,float32]
  Mat4x2f* = Mat[4,2,float32]
  Mat3x2f* = Mat[3,2,float32]
  Mat2x2f* = Mat[2,2,float32]
  # float64
  Mat4d*   = Mat[4,4,float64]
  Mat3d*   = Mat[3,3,float64]
  Mat2d*   = Mat[2,2,float64]
  Mat4x4d* = Mat[4,4,float64]
  Mat3x4d* = Mat[3,4,float64]
  Mat2x4d* = Mat[2,4,float64]
  Mat4x3d* = Mat[4,3,float64]
  Mat3x3d* = Mat[3,3,float64]
  Mat2x3d* = Mat[2,3,float64]
  Mat4x2d* = Mat[4,2,float64]
  Mat3x2d* = Mat[3,2,float64]
  Mat2x2d* = Mat[2,2,float64]
  # int32
  Mat4i*   = Mat[4,4,int32]
  Mat3i*   = Mat[3,3,int32]
  Mat2i*   = Mat[2,2,int32]
  Mat4x4i* = Mat[4,4,int32]
  Mat3x4i* = Mat[3,4,int32]
  Mat2x4i* = Mat[2,4,int32]
  Mat4x3i* = Mat[4,3,int32]
  Mat3x3i* = Mat[3,3,int32]
  Mat2x3i* = Mat[2,3,int32]
  Mat4x2i* = Mat[4,2,int32]
  Mat3x2i* = Mat[3,2,int32]
  Mat2x2i* = Mat[2,2,int32]
  # int64
  Mat4l*   = Mat[4,4,int64]
  Mat3l*   = Mat[3,3,int64]
  Mat2l*   = Mat[2,2,int64]
  Mat4x4l* = Mat[4,4,int64]
  Mat3x4l* = Mat[3,4,int64]
  Mat2x4l* = Mat[2,4,int64]
  Mat4x3l* = Mat[4,3,int64]
  Mat3x3l* = Mat[3,3,int64]
  Mat2x3l* = Mat[2,3,int64]
  Mat4x2l* = Mat[4,2,int64]
  Mat3x2l* = Mat[3,2,int64]
  Mat2x2l* = Mat[2,2,int64]
  # uint32
  Mat4ui*   = Mat[4,4,uint32]
  Mat3ui*   = Mat[3,3,uint32]
  Mat2ui*   = Mat[2,2,uint32]
  Mat4x4ui* = Mat[4,4,uint32]
  Mat3x4ui* = Mat[3,4,uint32]
  Mat2x4ui* = Mat[2,4,uint32]
  Mat4x3ui* = Mat[4,3,uint32]
  Mat3x3ui* = Mat[3,3,uint32]
  Mat2x3ui* = Mat[2,3,uint32]
  Mat4x2ui* = Mat[4,2,uint32]
  Mat3x2ui* = Mat[3,2,uint32]
  Mat2x2ui* = Mat[2,2,uint32]
  # uint64
  Mat4ul*   = Mat[4,4,uint64]
  Mat3ul*   = Mat[3,3,uint64]
  Mat2ul*   = Mat[2,2,uint64]
  Mat4x4ul* = Mat[4,4,uint64]
  Mat3x4ul* = Mat[3,4,uint64]
  Mat2x4ul* = Mat[2,4,uint64]
  Mat4x3ul* = Mat[4,3,uint64]
  Mat3x3ul* = Mat[3,3,uint64]
  Mat2x3ul* = Mat[2,3,uint64]
  Mat4x2ul* = Mat[4,2,uint64]
  Mat3x2ul* = Mat[3,2,uint64]
  Mat2x2ul* = Mat[2,2,uint64]

proc diag*[N,M,T](m : Mat[N,M,T]): Vec[min(N,M), T] =
  for i in 0 ..< min(N,M):
    result.arr[i] = m.arr[i].arr[i]

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

template mat2f*(                )  : untyped = mat2[float32](        )
template mat3f*(                )  : untyped = mat3[float32](        )
template mat4f*(                )  : untyped = mat4[float32](        )
template mat2f*(a:       untyped)  : untyped = mat2[float32](a       )
template mat3f*(a:       untyped)  : untyped = mat3[float32](a       )
template mat4f*(a:       untyped)  : untyped = mat4[float32](a       )
template mat2f*(a,b:     untyped)  : untyped = mat2[float32](a,b     )
template mat3f*(a,b,c:   untyped)  : untyped = mat3[float32](a,b,c   )
template mat4f*(a,b,c,d: untyped)  : untyped = mat4[float32](a,b,c,d )


template mat2d*(                )  : untyped = mat2[float64](        )
template mat3d*(                )  : untyped = mat3[float64](        )
template mat4d*(                )  : untyped = mat4[float64](        )
template mat2d*(a:       untyped)  : untyped = mat2[float64](a       )
template mat3d*(a:       untyped)  : untyped = mat3[float64](a       )
template mat4d*(a:       untyped)  : untyped = mat4[float64](a       )
template mat2d*(a,b:     untyped)  : untyped = mat2[float64](a,b     )
template mat3d*(a,b,c:   untyped)  : untyped = mat3[float64](a,b,c   )
template mat4d*(a,b,c,d: untyped)  : untyped = mat4[float64](a,b,c,d )


template mat2i*(                )  : untyped = mat2[int32](        )
template mat3i*(                )  : untyped = mat3[int32](        )
template mat4i*(                )  : untyped = mat4[int32](        )
template mat2i*(a:       untyped)  : untyped = mat2[int32](a       )
template mat3i*(a:       untyped)  : untyped = mat3[int32](a       )
template mat4i*(a:       untyped)  : untyped = mat4[int32](a       )
template mat2i*(a,b:     untyped)  : untyped = mat2[int32](a,b     )
template mat3i*(a,b,c:   untyped)  : untyped = mat3[int32](a,b,c   )
template mat4i*(a,b,c,d: untyped)  : untyped = mat4[int32](a,b,c,d )


template mat2l*(                )  : untyped = mat2[int64](        )
template mat3l*(                )  : untyped = mat3[int64](        )
template mat4l*(                )  : untyped = mat4[int64](        )
template mat2l*(a:       untyped)  : untyped = mat2[int64](a       )
template mat3l*(a:       untyped)  : untyped = mat3[int64](a       )
template mat4l*(a:       untyped)  : untyped = mat4[int64](a       )
template mat2l*(a,b:     untyped)  : untyped = mat2[int64](a,b     )
template mat3l*(a,b,c:   untyped)  : untyped = mat3[int64](a,b,c   )
template mat4l*(a,b,c,d: untyped)  : untyped = mat4[int64](a,b,c,d )


template mat2ui*(                )  : untyped = mat2[uint32](        )
template mat3ui*(                )  : untyped = mat3[uint32](        )
template mat4ui*(                )  : untyped = mat4[uint32](        )
template mat2ui*(a:       untyped)  : untyped = mat2[uint32](a       )
template mat3ui*(a:       untyped)  : untyped = mat3[uint32](a       )
template mat4ui*(a:       untyped)  : untyped = mat4[uint32](a       )
template mat2ui*(a,b:     untyped)  : untyped = mat2[uint32](a,b     )
template mat3ui*(a,b,c:   untyped)  : untyped = mat3[uint32](a,b,c   )
template mat4ui*(a,b,c,d: untyped)  : untyped = mat4[uint32](a,b,c,d )


template mat2ul*(                )  : untyped = mat2[uint64](        )
template mat3ul*(                )  : untyped = mat3[uint64](        )
template mat4ul*(                )  : untyped = mat4[uint64](        )
template mat2ul*(a:       untyped)  : untyped = mat2[uint64](a       )
template mat3ul*(a:       untyped)  : untyped = mat3[uint64](a       )
template mat4ul*(a:       untyped)  : untyped = mat4[uint64](a       )
template mat2ul*(a,b:     untyped)  : untyped = mat2[uint64](a,b     )
template mat3ul*(a,b,c:   untyped)  : untyped = mat3[uint64](a,b,c   )
template mat4ul*(a,b,c,d: untyped)  : untyped = mat4[uint64](a,b,c,d )

#[

# generic
template namedConstructors(postfix: untyped, Type : typedesc): untyped =
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


## <end>


proc mat4f*() : Mat4f =
  for i in 0 .. 3:
    result.arr[i].arr[i] = 1

proc mat3f*() : Mat3f =
  for i in 0 .. 2:
    result.arr[i].arr[i] = 1

proc mat2f*() : Mat2f =
  for i in 0 .. 1:
    result.arr[i].arr[i] = 1

proc mat4d*() : Mat4d =
  for i in 0 .. 3:
    result.arr[i].arr[i] = 1

proc mat3d*() : Mat3d =
  for i in 0 .. 2:
    result.arr[i].arr[i] = 1

proc mat2d*() : Mat2d =
  for i in 0 .. 1:
    result.arr[i].arr[i] = 1

proc mat4i*() : Mat4i =
  for i in 0 .. 3:
    result.arr[i].arr[i] = 1

proc mat3i*() : Mat3i =
  for i in 0 .. 2:
    result.arr[i].arr[i] = 1

proc mat2i*() : Mat2i =
  for i in 0 .. 1:
    result.arr[i].arr[i] = 1

proc mat4l*() : Mat4l =
  for i in 0 .. 3:
    result.arr[i].arr[i] = 1

proc mat3l*() : Mat3l =
  for i in 0 .. 2:
    result.arr[i].arr[i] = 1

proc mat2l*() : Mat2l =
  for i in 0 .. 1:
    result.arr[i].arr[i] = 1

proc mat4f*(a,b,c,d: Vec4f) : Mat4f =
  result.arr = [a,b,c,d]

proc mat3f*(a,b,c: Vec3f) : Mat3f =
  result.arr = [a,b,c]

proc mat2f*(a,b: Vec2f) : Mat2f =
  result.arr = [a,b]

proc mat4d*(a,b,c,d: Vec4d) : Mat4d =
  result.arr = [a,b,c,d]

proc mat3d*(a,b,c: Vec3d) : Mat3d =
  result.arr = [a,b,c]

proc mat2d*(a,b: Vec2d) : Mat2d =
  result.arr = [a,b]

proc mat4i*(a,b,c,d: Vec4i) : Mat4i =
  result.arr = [a,b,c,d]

proc mat3i*(a,b,c: Vec3i) : Mat3i =
  result.arr = [a,b,c]

proc mat2i*(a,b: Vec2i) : Mat2i =
  result.arr = [a,b]

proc mat4l*(a,b,c,d: Vec4l) : Mat4l =
  result.arr = [a,b,c,d]

proc mat3l*(a,b,c: Vec3l) : Mat3l =
  result.arr = [a,b,c]

proc mat2l*(a,b: Vec2l) : Mat2l =
  result.arr = [a,b]
]#

#proc `==`(m1,m2: Mat): Mat =

#diagonalConstructors(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE)
#matrixComparison(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE)

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

template numCols*[N,M,T](t : typedesc[Mat[N,M,T]]): int = N
template numRows*[N,M,T](t : typedesc[Mat[N,M,T]]): int = M

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

proc mix*[N,M,T](v1,v2: Mat[N,M,T]; a: T): Mat[N,M,T] =
  v1 * (1 - a) + v2 * a

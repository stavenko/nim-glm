#import strutils
#import sequtils
import macros, math

export math.Pi

# this is a dirty hack to have integer division behave like in C/C++/glsl etc.
# don't export functions, maybe disable
template `/`(a,b: int32): int32 = a div b
template `/`(a,b: int64): int64 = a div b
template `/`(a,b: int): int = a div b
proc `/=`(a: var SomeInteger; b: SomeInteger): void =
  a = a div b

##Vector module contains all types and functions to manipulate vectors
type
  Vec*[N : static[int], T] = object
    arr*: array[N, T]

type
  Vec4*[T] = Vec[4,T]
  Vec3*[T] = Vec[3,T]
  Vec2*[T] = Vec[2,T]

type
  Vec4f*  = Vec[4, float32]
  Vec3f*  = Vec[3, float32]
  Vec2f*  = Vec[2, float32]
  Vec4d*  = Vec[4, float64]
  Vec3d*  = Vec[3, float64]
  Vec2d*  = Vec[2, float64]
  Vec4i*  = Vec[4, int32]
  Vec3i*  = Vec[3, int32]
  Vec2i*  = Vec[2, int32]
  Vec4l*  = Vec[4, int64]
  Vec3l*  = Vec[3, int64]
  Vec2l*  = Vec[2, int64]
  Vec4ui*  = Vec[4, uint32]
  Vec3ui*  = Vec[3, uint32]
  Vec2ui*  = Vec[2, uint32]
  Vec4ul*  = Vec[4, uint64]
  Vec3ul*  = Vec[3, uint64]
  Vec2ul*  = Vec[2, uint64]
  Vec4b*  = Vec[4, bool]
  Vec3b*  = Vec[3, bool]
  Vec2b*  = Vec[2, bool]

proc `$`*(v: Vec) : string =
  result = "["
  for i, val in v.arr:
    result &= $val
    if i != v.N - 1:
      result &= "  "
  result &= "]"

proc spaces(num: int): string =
  result = newString(num)
  for c in result.mitems:
    c = ' '

proc alignRight*[N,T](v: array[N, T]) : array[N,string] =
  var maxLen = 0
  for i, val in v:
    result[i] = $val
    maxLen = max(maxLen, result[i].len)
  for i, str in result.mpairs:
    str.insert(spaces(maxLen - str.len))

proc alignLeft*[N,T](v: array[N, T]) : array[N,string] =
  var maxLen = 0
  for i, val in v:
    result[i] = $val
    maxLen = max(maxLen, result[i].len)
  for i, str in result.mpairs:
    str.add(spaces(maxLen - str.len))

proc alignChar*[N,T](v: array[N, T]; c: char) : array[N,string] =
  for i, val in v:
    result[i] = $val

  var lenLeft  : array[N, int]
  var maxLenLeft = 0
  var lenRight : array[N, int]
  var maxLenRight = 0

  for i, str in result:
    let index = str.find(c)
    let length = str.len
    lenLeft[i]  = index
    maxLenLeft = max(maxLenLeft, lenLeft[i])
    lenRight[i] = length - index - 1
    maxLenRight = max(maxLenRight, lenRight[i])

  for i, str in result.mpairs:
    str.insert(spaces(maxLenLeft  - lenLeft[i]))
    str.add(   spaces(maxLenRight - lenRight[i]))

proc columnFormat*[N,T](v: Vec[N, T]) : array[N,string] =
  when T is SomeInteger:
    result = v.arr.alignRight
  elif T is SomeReal:
    result = v.arr.alignChar('.')
  else:
    result = v.arr.alignLeft

template mathPerComponent(op: untyped): untyped =
  # TODO this is a good place for simd optimization

  proc op*[N,T](v,u: Vec[N,T]): Vec[N,T] {.inline.} =
    for ii in 0 ..< N:
      result.arr[ii] = op(v.arr[ii], u.arr[ii])

  proc op*[N,T](v: Vec[N,T]; val: T): Vec[N,T] {.inline.} =
    for ii in 0 ..< N:
      result.arr[ii] = op(v.arr[ii], val)

  proc op*[N,T](val: T; v: Vec[N,T]): Vec[N,T] {.inline.} =
    for ii in 0 ..< N:
      result.arr[ii] = op(val, v.arr[ii])

mathPerComponent(`+`)
mathPerComponent(`-`)
mathPerComponent(`/`)
mathPerComponent(`*`)
mathPerComponent(`div`)

template mathInpl(opName): untyped =
  proc opName*[N,T](v: var Vec[N,T]; u: Vec[N,T]): void =
    for ii in 0 ..< N:
      opName(v.arr[ii], u.arr[ii])

  proc opName*[N,T](v: var Vec[N,T]; x: T): void =
    for ii in 0 ..< N:
      opName(v.arr[ii], x)

mathInpl(`+=`)
mathInpl(`-=`)
mathInpl(`*=`)
mathInpl(`/=`)

# unary operators
proc `-`*[N,T](v: Vec[N,T]): Vec[N,T] =
  for i in 0 ..< N:
    result.arr[i] = -v.arr[i]

proc `+`*[N,T](v: Vec[N,T]): Vec[N,T] = v

proc `[]=`*[N,T](v:var Vec[N,T]; ix:int; c:T): void {.inline.} =
    v.arr[ix] = c
proc `[]`*[N,T](v: Vec[N,T]; ix: int): T {.inline.} =
  v.arr[ix]
proc `[]`*[N,T](v: var Vec[N,T]; ix: int): var T {.inline.} =
  v.arr[ix]

#########################
# constructor functions #
#########################

proc vec4*[T](x,y,z,w:T)         : Vec4[T] {.inline.} = Vec4[T](arr: [  x,   y,   z,   w])
proc vec4*[T](v:Vec3[T],w:T)     : Vec4[T] {.inline.} = Vec4[T](arr: [v.x, v.y, v.z,   w])
proc vec4*[T](x:T,v:Vec3[T])     : Vec4[T] {.inline.} = Vec4[T](arr: [  x, v.x, v.y, v.z])
proc vec4*[T](a,b:Vec2[T])       : Vec4[T] {.inline.} = Vec4[T](arr: [a.x, a.y, b.x, b.y])
proc vec4*[T](v:Vec2[T],z,w:T)   : Vec4[T] {.inline.} = Vec4[T](arr: [v.x, v.y,   z,   w])
proc vec4*[T](x:T,v:Vec2[T],w:T) : Vec4[T] {.inline.} = Vec4[T](arr: [  x, v.x, v.y,   w])
proc vec4*[T](x,y:T,v:Vec2[T])   : Vec4[T] {.inline.} = Vec4[T](arr: [  x,   y, v.x, v.y])
proc vec4*[T](x:T)               : Vec4[T] {.inline.} = Vec4[T](arr: [  x,   x,   x,   x])

proc vec3*[T](x,y,z: T)      : Vec3[T] {.inline.} = Vec3[T](arr: [  x,   y,   z])
proc vec3*[T](v:Vec2[T],z:T) : Vec3[T] {.inline.} = Vec3[T](arr: [v.x, v.y,   z])
proc vec3*[T](x:T,v:Vec2[T]) : Vec3[T] {.inline.} = Vec3[T](arr: [  x, v.x, v.y])
proc vec3*[T](x:T)           : Vec3[T] {.inline.} = Vec3[T](arr: [  x,   x,   x])

proc vec2*[T](x,y:T) : Vec2[T] {.inline.} = Vec2[T](arr: [x,y])
proc vec2*[T](x:T)   : Vec2[T] {.inline.} = Vec2[T](arr: [x,x])

proc subVec[N,T](v: var Vec[N,T]; offset, length: static[int]): var Vec[length,T] {.inline.} =
  cast[ptr Vec[length, T]](v.arr[offset].addr)[]

proc growingIndices(indices: varargs[int]): bool =
  ## returns true when every argument is bigger than all previous arguments
  for i in 1 .. indices.high:
    if indices[i-1] >= indices[i]:
      return false
  return true

proc continuousIndices(indices: varargs[int]): bool =
  for i in 1 .. indices.high:
    if indices[i-1] != indices[i]-1:
      return false
  return true

proc head(node: NimNode): NimNode {.compileTime.} = node[0]

proc swizzleMethods(indices: varargs[int]) : seq[NimNode] {.compileTime.}=
  result.newSeq(0)

  const chars = "xyzw"

  var name = ""
  for idx in indices:
    name.add chars[idx]

  let getIdent = ident(name)
  let setIdent = ident(name & '=')

  if indices.len > 1:

    let bracket = nnkBracket.newTree

    let Nlit = newLit(indices.len)
    let v = genSym(nskParam, "v")

    for idx in indices:
      let lit = newLit(idx)
      bracket.add head quote do:
        `v`.arr[`lit`]

    result.add head quote do:
      proc `getIdent`*[N,T](`v`: Vec[N,T]): Vec[`Nlit`,T] {.inline.} =
        Vec[`Nlit`,T](arr: `bracket`)

    #if continuousIndices(indices):
    #  echo result.back.repr

    if continuousIndices(indices):
      #echo result.back.repr

      let offsetLit = newLit(indices[0])
      let lengthLit = newLit(indices.len)
      result.add head quote do:
        proc `getIdent`*[N,T](v: var Vec[N,T]): var Vec[`Nlit`,T] {.inline.} =
          v.subVec(`offsetLit`, `lengthLit`)


    if growingIndices(indices):
      let N2lit = newLit(indices.len)
      let v1 = genSym(nskParam, "v1")
      let v2 = genSym(nskParam, "v2")

      let assignments = newStmtList()
      for i,idx in indices:
        let litL = newLit(idx)
        let litR = newLit(i)
        assignments.add head quote do:
          `v1`.arr[`litL`] = `v2`.arr[`litR`]

      result.add head quote do:
        proc `setIdent`*[N,T](`v1`: var Vec[N,T]; `v2`: Vec[`N2lit`,T]): void =
          `assignments`

  else:
    let lit = newLit(indices[0])
    result.add quote do:
      proc `getIdent`*[N,T](v: Vec[N,T]): T {.inline.} =
        v.arr[`lit`]

      proc `getIdent`*[N,T](v: var Vec[N,T]): var T {.inline.} =
        v.arr[`lit`]

      proc `setIdent`*[N,T](v: var Vec[N,T]; val: T): void {.inline.} =
        v.arr[`lit`] = val



macro genSwizzleOps*(): untyped =
  result = newStmtList()
  for i in 0 .. 3:
    result.add swizzleMethods(i)
    for j in 0 .. 3:
      result.add swizzleMethods(i,j)
      for k in 0 .. 3:
        result.add swizzleMethods(i,j,k)
        for m in 0 .. 3:
          result.add swizzleMethods(i,j,k,m)

genSwizzleOps()

proc caddr*[N,T](v:var Vec[N,T]): ptr T {.inline.}=
  ## Address getter to pass vector to native-C openGL functions as pointers
  v.arr[0].addr


####################################
# Angle and Trigonometry Functions #
####################################

template foreachImpl(fun: untyped): untyped =
  proc fun*[N,T](v: Vec[N,T]): Vec[N,T] =
    for i in 0 ..< N:
      result.arr[i] = fun(v.arr[i])

template foreachZipImpl(fun: untyped): untyped =
  proc fun*[N,T](v1,v2: Vec[N,T]): Vec[N,T] =
    for i in 0 ..< N:
      result.arr[i] = fun(v1.arr[i], v2.arr[i])

export math.sin, math.cos, math.tan

foreachImpl(sin)
foreachImpl(cos)
foreachImpl(tan)

export math.arcsin, math.arccos, math.arctan

foreachImpl(arcsin)
foreachImpl(arccos)
foreachImpl(arctan)

export math.sinh, math.cosh, math.tanh

foreachImpl(sinh)
foreachImpl(cosh)
foreachImpl(tanh)

proc radians*(v : SomeReal): SomeReal {.inline.} =
  v * math.Pi / 180

proc degrees*(v : SomeReal): SomeReal {.inline.} =
  v * 180 / math.Pi

foreachImpl(radians)
foreachImpl(degrees)

#########################
# Exponential Functions #
#########################

export math.pow

foreachZipImpl(pow)

proc exp2*(x: SomeReal): SomeReal {.inline.} = math.pow(2,x)
proc inversesqrt*(x: SomeReal): SomeReal {.inline.} = 1 / sqrt(x)
export math.exp, math.ln, math.log2, math.sqrt

foreachImpl(exp2)
foreachImpl(inversesqrt)
foreachImpl(exp)
foreachImpl(ln)
foreachImpl(log2)
foreachImpl(sqrt)

####################
# common functions #
####################

export math.ceil, math.floor, math.round

proc abs*[N,T](v : Vec[N,T]) : Vec[N,T] =
  for i in 0 ..< N:
    result.arr[i] = abs(v.arr[i])

proc ceil*[N,T](v: Vec[N,T]): Vec[N,T] =
  for i in 0 ..< N:
    result.arr[i] = ceil(v.arr[i])

proc round*[N,T](v: Vec[N,T]): Vec[N,T] =
  for i in 0 ..< N:
    result.arr[i] = round(v.arr[i])

proc clamp*[N,T](arg, minVal, maxVal: Vec[N,T]): Vec[N,T] =
  for i in 0 ..< N:
    result.arr[i] = clamp(arg.arr[i], minVal.arr[i], maxVal.arr[i])

proc clamp*[N,T](arg: Vec[N,T]; minVal, maxVal: T): Vec[N,T] =
  for i in 0 ..< N:
    result.arr[i] = clamp(arg.arr[i], minVal, maxVal)

proc floor*[N,T](v : Vec[N,T]) : Vec[N,T] =
  for i in 0 ..< N:
    result.arr[i] = floor(v.arr[i])

proc fract*[T](v : T): T =
  v - floor(v)

proc fract*[N,T](v : Vec[N,T]) : Vec[N,T] =
  for i in 0 ..< N:
    result.arr[i] = fract(v.arr[i])

proc max*[N,T](v1,v2: Vec[N,T]): Vec[N,T] =
  for i in 0 ..< N:
    result.arr[i] = max(v1.arr[i], v2.arr[i])

proc max*[N,T](v: Vec[N,T]; val: T): Vec[N,T] =
  for i in 0 ..< N:
    result.arr[i] = max(v.arr[i], val)

proc max*[N,T](val: T; v: Vec[N,T]): Vec[N,T] =
  for i in 0 ..< N:
    result.arr[i] = max(val, v.arr[i])

proc min*[N,T](v1,v2: Vec[N,T]): Vec[N,T] =
  for i in 0 ..< N:
    result.arr[i] = min(v1.arr[i], v2.arr[i])

proc min*[N,T](v: Vec[N,T]; val: T): Vec[N,T] =
  for i in 0 ..< N:
    result.arr[i] = min(v.arr[i], val)

proc min*[N,T](val: T; v: Vec[N,T]): Vec[N,T] =
  for i in 0 ..< N:
    result.arr[i] = min(val, v.arr[i])

proc mix*[T: SomeNumber](x,y,a: T): T =
  x * (1 - a) + y * a

proc mix*[N,T](v1,v2: Vec[N,T]; a: T): Vec[N,T] =
  v1 * (1 - a) + v2 * a

proc mix*[N,T](v1,v2,a: Vec[N,T]): Vec[N,T] =
  # untested
  v1 * (1 - a) + v2 * a

proc fmod*(x,y: SomeReal): SomeReal =
  y * fract(x / y)

proc fmod*[N,T](v1,v2: Vec[N,T]): Vec[N,T] =
  # untested
  v2 * fract(v1 / v2)

proc fmod*[N,T](v: Vec[N,T]; val: T): Vec[N,T] =
  # untested
  val * fract(v / val)

proc sign*[T](x: T): T =
  T(x > 0) - T(x < 0)

proc sign*[N,T](v: Vec[N,T]): Vec[N,T] =
  # untested
  for i in 0 ..< N:
    result.arr[i] = sign(v.arr[i])

proc smoothstep*(edge0,edge1,x: SomeReal): SomeReal =
  ## performs smooth Hermite interpolation between 0 and 1 when edge0 < x < edge1.
  ## This is useful in cases where a threshold function with a smooth transition is desired
  # untested
  let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
  return t * t * (3 - 2 * t)

proc smoothstep*[N,T](edge0,edge1,x: Vec[N,T]): Vec[N,T] =
  ## performs smooth Hermite interpolation between 0 and 1 when edge0 < x < edge1.
  ## This is useful in cases where a threshold function with a smooth transition is desired
  # untested
  let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
  return t * t * (3 - 2 * t)

proc smoothstep*[N,T](edge0,edge1: T; x: Vec[N,T]): Vec[N,T] =
  ## performs smooth Hermite interpolation between 0 and 1 when edge0 < x < edge1.
  ## This is useful in cases where a threshold function with a smooth transition is desired
  # untested
  let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
  return t * t * (3 - 2 * t)

proc step*[T](edge,x: T): T =
  return T(x >= edge)

proc step*[N,T](edge,x: Vec[N,T]): Vec[N,T] =
  ## For element i of the return value, 0.0 is returned if x[i] < edge[i], and 1.0 is returned otherwise
  for i in 0 ..< N:
    result.arr[i] = T(x.arr[i] >= edge.arr[i])

proc step*[N,T](edge: T; x: Vec[N,T]): Vec[N,T] =
  ## For element i of the return value, 0.0 is returned if x[i] < edge, and 1.0 is returned otherwise
  for i in 0 ..< N:
    result.arr[i] = T(x.arr[i] >= edge)


#######################
# Geometric Functions #
#######################

proc dot*[N,T](u,v: Vec[N,T]): T {. inline .} =
  # TODO this really should have some simd optimization
  # matrix multiplication is based on this
  for i in 0 ..< N:
    result += u[i] * v[i]

proc length2*(v: Vec): auto = dot(v,v)
proc length*(v: Vec): auto = sqrt(dot(v,v))

proc cross*[T](v1,v2:Vec[3,T]): Vec[3,T] =
  v1.yzx * v2.zxy - v1.zxy * v2.yzx

proc distance*[N,T](v1,v2: Vec[N,T]): Vec[N,T] = length(v2 - v1)


proc faceforward*[N,T](n,i,nref: Vec[N,T]): Vec[N,T] =
  ## return a vector pointing in the same direction as another
  n * (float(dot(nref, i) < 0) * 2 - 1)

proc normalize*[N,T](v: Vec[N,T]): Vec[N,T] = v * (T(1) / v.length)

proc reflect*[N,T](i,n: Vec[N,T]): Vec[N,T] =
  ## For a given incident vector ``i`` and surface normal ``n`` reflect returns the reflection direction
  i - 2 * dot(n,i) * n

proc refract*[N,T](i,n: Vec[N,T]; eta: T): Vec[N,T] =
  # For a given incident vector ``i``, surface normal ``n`` and ratio of indices of refraction, ``eta``, refract returns the refraction vector.
  let k = 1 - eta * eta * (1 - dot(n, i) * dot(n, i));
  if k >= 0.0:
    result = eta * i - (eta * dot(n, i) + sqrt(k)) * n;

###################
# more type names #
###################

type
  Vec4u8* = Vec[4, uint8]

proc vec4f*(x,y,z,w:float32)             : Vec4f {.inline.} = Vec4f(arr: [  x,   y,   z,   w])
proc vec4f*(v:Vec3f,w:float32)           : Vec4f {.inline.} = Vec4f(arr: [v.x, v.y, v.z,   w])
proc vec4f*(x:float32,v:Vec3f)           : Vec4f {.inline.} = Vec4f(arr: [  x, v.x, v.y, v.z])
proc vec4f*(a,b:Vec2f)                   : Vec4f {.inline.} = Vec4f(arr: [a.x, a.y, b.x, b.y])
proc vec4f*(v:Vec2f,z,w:float32)         : Vec4f {.inline.} = Vec4f(arr: [v.x, v.y,   z,   w])
proc vec4f*(x:float32,v:Vec2f,w:float32) : Vec4f {.inline.} = Vec4f(arr: [  x, v.x, v.y,   w])
proc vec4f*(x,y:float32,v:Vec2f)         : Vec4f {.inline.} = Vec4f(arr: [  x,   y, v.x, v.y])
proc vec4f*(x:float32)                   : Vec4f {.inline.} = Vec4f(arr: [  x,   x,   x,   x])

proc vec3f*(x,y,z:   float32)  : Vec3f {.inline.} = Vec3f(arr: [  x,   y,   z])
proc vec3f*(v:Vec2f,z:float32) : Vec3f {.inline.} = Vec3f(arr: [v.x, v.y,   z])
proc vec3f*(x:float32,v:Vec2f) : Vec3f {.inline.} = Vec3f(arr: [  x, v.x, v.y])
proc vec3f*(x:float32)         : Vec3f {.inline.} = Vec3f(arr: [  x,   x,   x])

proc vec2f*(x,y:float32) : Vec2f {.inline.} = Vec2f(arr: [x,y])
proc vec2f*(x:float32)   : Vec2f {.inline.} = Vec2f(arr: [x,x])

proc vec4f*(a:array[0..3, float32]) : Vec4f {.inline.} = Vec4f(arr: [a[0], a[1], a[2], a[3]])
proc vec3f*(a:array[0..2, float32]) : Vec3f {.inline.} = Vec3f(arr: [a[0], a[1], a[2]])
proc vec2f*(a:array[0..1, float32]) : Vec2f {.inline.} = Vec2f(arr: [a[0], a[1]])

proc vec4d*(x,y,z,w:float64)             : Vec4d {.inline.} = Vec4d(arr: [  x,   y,   z,   w])
proc vec4d*(v:Vec3d,w:float64)           : Vec4d {.inline.} = Vec4d(arr: [v.x, v.y, v.z,   w])
proc vec4d*(x:float64,v:Vec3d)           : Vec4d {.inline.} = Vec4d(arr: [  x, v.x, v.y, v.z])
proc vec4d*(a,b:Vec2d)                   : Vec4d {.inline.} = Vec4d(arr: [a.x, a.y, b.x, b.y])
proc vec4d*(v:Vec2d,z,w:float64)         : Vec4d {.inline.} = Vec4d(arr: [v.x, v.y,   z,   w])
proc vec4d*(x:float64,v:Vec2d,w:float64) : Vec4d {.inline.} = Vec4d(arr: [  x, v.x, v.y,   w])
proc vec4d*(x,y:float64,v:Vec2d)         : Vec4d {.inline.} = Vec4d(arr: [  x,   y, v.x, v.y])
proc vec4d*(x:float64)                   : Vec4d {.inline.} = Vec4d(arr: [  x,   x,   x,   x])

proc vec3d*(x,y,z:   float64)  : Vec3d {.inline.} = Vec3d(arr: [  x,   y,   z])
proc vec3d*(v:Vec2d,z:float64) : Vec3d {.inline.} = Vec3d(arr: [v.x, v.y,   z])
proc vec3d*(x:float64,v:Vec2d) : Vec3d {.inline.} = Vec3d(arr: [  x, v.x, v.y])
proc vec3d*(x:float64)         : Vec3d {.inline.} = Vec3d(arr: [  x,   x,   x])

proc vec2d*(x,y:float64) : Vec2d {.inline.} = Vec2d(arr: [x,y])
proc vec2d*(x:float64)   : Vec2d {.inline.} = Vec2d(arr: [x,x])


proc vec4i*(x,y,z,w:int32)             : Vec4i {.inline.} = Vec4i(arr: [  x,   y,   z,   w])
proc vec4i*(v:Vec3i; w:int32)          : Vec4i {.inline.} = Vec4i(arr: [v.x, v.y, v.z,   w])
proc vec4i*(x:int32; v:Vec3i)          : Vec4i {.inline.} = Vec4i(arr: [  x, v.x, v.y, v.z])
proc vec4i*(a,b:Vec2i)                 : Vec4i {.inline.} = Vec4i(arr: [a.x, a.y, b.x, b.y])
proc vec4i*(v:Vec2i; z,w:int32)        : Vec4i {.inline.} = Vec4i(arr: [v.x, v.y,   z,   w])
proc vec4i*(x:int32; v:Vec2i; w:int32) : Vec4i {.inline.} = Vec4i(arr: [  x, v.x, v.y,   w])
proc vec4i*(x,y:int32; v:Vec2i)        : Vec4i {.inline.} = Vec4i(arr: [  x,   y, v.x, v.y])
proc vec4i*(x:int32)                   : Vec4i {.inline.} = Vec4i(arr: [  x,   x,   x,   x])

proc vec3i*(x,y,z:int32)      : Vec3i {.inline.} = Vec3i(arr: [  x,   y,   z])
proc vec3i*(v:Vec2i; z:int32) : Vec3i {.inline.} = Vec3i(arr: [v.x, v.y,   z])
proc vec3i*(x:int32; v:Vec2i) : Vec3i {.inline.} = Vec3i(arr: [  x, v.x, v.y])
proc vec3i*(x:int32)          : Vec3i {.inline.} = Vec3i(arr: [  x,   x,   x])

proc vec2i*(x,y:int32) : Vec2i {.inline.} = Vec2i(arr: [x,y])
proc vec2i*(x:int32)   : Vec2i {.inline.} = Vec2i(arr: [x,x])

proc vec4i*(a:array[0..3, int32]) : Vec4i {.inline.} = Vec4i(arr: [a[0], a[1], a[2], a[3]])
proc vec3i*(a:array[0..2, int32]) : Vec3i {.inline.} = Vec3i(arr: [a[0], a[1], a[2]])
proc vec2i*(a:array[0..1, int32]) : Vec2i {.inline.} = Vec2i(arr: [a[0], a[1]])

proc vec4l*(x,y,z,w:int64)             : Vec4l {.inline.} = Vec4l(arr: [  x,   y,   z,   w])
proc vec4l*(v:Vec3l; w:int64)          : Vec4l {.inline.} = Vec4l(arr: [v.x, v.y, v.z,   w])
proc vec4l*(x:int64; v:Vec3l)          : Vec4l {.inline.} = Vec4l(arr: [  x, v.x, v.y, v.z])
proc vec4l*(a,b:Vec2l)                 : Vec4l {.inline.} = Vec4l(arr: [a.x, a.y, b.x, b.y])
proc vec4l*(v:Vec2l; z,w:int64)        : Vec4l {.inline.} = Vec4l(arr: [v.x, v.y,   z,   w])
proc vec4l*(x:int64; v:Vec2l; w:int64) : Vec4l {.inline.} = Vec4l(arr: [  x, v.x, v.y,   w])
proc vec4l*(x,y:int64; v:Vec2l)        : Vec4l {.inline.} = Vec4l(arr: [  x,   y, v.x, v.y])
proc vec4l*(x:int64)                   : Vec4l {.inline.} = Vec4l(arr: [  x,   x,   x,   x])

proc vec3l*(x,y,z:int64)      : Vec3l {.inline.} = Vec3l(arr: [  x,   y,   z])
proc vec3l*(v:Vec2l; z:int64) : Vec3l {.inline.} = Vec3l(arr: [v.x, v.y,   z])
proc vec3l*(x:int64; v:Vec2l) : Vec3l {.inline.} = Vec3l(arr: [  x, v.x, v.y])
proc vec3l*(x:int64)          : Vec3l {.inline.} = Vec3l(arr: [  x,   x,   x])

proc vec2l*(x,y:int64) : Vec2l {.inline.} = Vec2l(arr: [x,y])
proc vec2l*(x:int64)   : Vec2l {.inline.} = Vec2l(arr: [x,x])

proc vec4l*(a:array[0..3, int64]) : Vec4l {.inline.} = Vec4l(arr: [a[0], a[1], a[2], a[3]])
proc vec3l*(a:array[0..2, int64]) : Vec3l {.inline.} = Vec3l(arr: [a[0], a[1], a[2]])
proc vec2l*(a:array[0..1, int64]) : Vec2l {.inline.} = Vec2l(arr: [a[0], a[1]])


proc vec4ui*(x,y,z,w:uint32)             : Vec4ui {.inline.} = Vec4ui(arr: [  x,   y,   z,   w])
proc vec4ui*(v:Vec3ui; w:uint32)          : Vec4ui {.inline.} = Vec4ui(arr: [v.x, v.y, v.z,   w])
proc vec4ui*(x:uint32; v:Vec3ui)          : Vec4ui {.inline.} = Vec4ui(arr: [  x, v.x, v.y, v.z])
proc vec4ui*(a,b:Vec2ui)                 : Vec4ui {.inline.} = Vec4ui(arr: [a.x, a.y, b.x, b.y])
proc vec4ui*(v:Vec2ui; z,w:uint32)        : Vec4ui {.inline.} = Vec4ui(arr: [v.x, v.y,   z,   w])
proc vec4ui*(x:uint32; v:Vec2ui; w:uint32) : Vec4ui {.inline.} = Vec4ui(arr: [  x, v.x, v.y,   w])
proc vec4ui*(x,y:uint32; v:Vec2ui)        : Vec4ui {.inline.} = Vec4ui(arr: [  x,   y, v.x, v.y])
proc vec4ui*(x:uint32)                   : Vec4ui {.inline.} = Vec4ui(arr: [  x,   x,   x,   x])

proc vec3ui*(x,y,z:uint32)      : Vec3ui {.inline.} = Vec3ui(arr: [  x,   y,   z])
proc vec3ui*(v:Vec2ui; z:uint32) : Vec3ui {.inline.} = Vec3ui(arr: [v.x, v.y,   z])
proc vec3ui*(x:uint32; v:Vec2ui) : Vec3ui {.inline.} = Vec3ui(arr: [  x, v.x, v.y])
proc vec3ui*(x:uint32)          : Vec3ui {.inline.} = Vec3ui(arr: [  x,   x,   x])

proc vec2ui*(x,y:uint32) : Vec2ui {.inline.} = Vec2ui(arr: [x,y])
proc vec2ui*(x:uint32)   : Vec2ui {.inline.} = Vec2ui(arr: [x,x])

proc vec4ui*(a:array[0..3, uint32]) : Vec4ui {.inline.} = Vec4ui(arr: [a[0], a[1], a[2], a[3]])
proc vec3ui*(a:array[0..2, uint32]) : Vec3ui {.inline.} = Vec3ui(arr: [a[0], a[1], a[2]])
proc vec2ui*(a:array[0..1, uint32]) : Vec2ui {.inline.} = Vec2ui(arr: [a[0], a[1]])

proc vec4ul*(x,y,z,w:uint64)             : Vec4ul {.inline.} = Vec4ul(arr: [  x,   y,   z,   w])
proc vec4ul*(v:Vec3ul; w:uint64)          : Vec4ul {.inline.} = Vec4ul(arr: [v.x, v.y, v.z,   w])
proc vec4ul*(x:uint64; v:Vec3ul)          : Vec4ul {.inline.} = Vec4ul(arr: [  x, v.x, v.y, v.z])
proc vec4ul*(a,b:Vec2ul)                 : Vec4ul {.inline.} = Vec4ul(arr: [a.x, a.y, b.x, b.y])
proc vec4ul*(v:Vec2ul; z,w:uint64)        : Vec4ul {.inline.} = Vec4ul(arr: [v.x, v.y,   z,   w])
proc vec4ul*(x:uint64; v:Vec2ul; w:uint64) : Vec4ul {.inline.} = Vec4ul(arr: [  x, v.x, v.y,   w])
proc vec4ul*(x,y:uint64; v:Vec2ul)        : Vec4ul {.inline.} = Vec4ul(arr: [  x,   y, v.x, v.y])
proc vec4ul*(x:uint64)                   : Vec4ul {.inline.} = Vec4ul(arr: [  x,   x,   x,   x])

proc vec3ul*(x,y,z:uint64)      : Vec3ul {.inline.} = Vec3ul(arr: [  x,   y,   z])
proc vec3ul*(v:Vec2ul; z:uint64) : Vec3ul {.inline.} = Vec3ul(arr: [v.x, v.y,   z])
proc vec3ul*(x:uint64; v:Vec2ul) : Vec3ul {.inline.} = Vec3ul(arr: [  x, v.x, v.y])
proc vec3ul*(x:uint64)          : Vec3ul {.inline.} = Vec3ul(arr: [  x,   x,   x])

proc vec2ul*(x,y:uint64) : Vec2ul {.inline.} = Vec2ul(arr: [x,y])
proc vec2ul*(x:uint64)   : Vec2ul {.inline.} = Vec2ul(arr: [x,x])

proc vec4ul*(a:array[0..3, uint64]) : Vec4ul {.inline.} = Vec4ul(arr: [a[0], a[1], a[2], a[3]])
proc vec3ul*(a:array[0..2, uint64]) : Vec3ul {.inline.} = Vec3ul(arr: [a[0], a[1], a[2]])
proc vec2ul*(a:array[0..1, uint64]) : Vec2ul {.inline.} = Vec2ul(arr: [a[0], a[1]])


proc vec4b*(x,y,z,w:bool)            : Vec4b {.inline.} = Vec4b(arr: [  x,   y,   z,   w])
proc vec4b*(v:Vec3b; w:bool)         : Vec4b {.inline.} = Vec4b(arr: [v.x, v.y, v.z,   w])
proc vec4b*(x:bool; v:Vec3b)         : Vec4b {.inline.} = Vec4b(arr: [  x, v.x, v.y, v.z])
proc vec4b*(a,b:Vec2b)               : Vec4b {.inline.} = Vec4b(arr: [a.x, a.y, b.x, b.y])
proc vec4b*(v:Vec2b; z,w:bool)       : Vec4b {.inline.} = Vec4b(arr: [v.x, v.y,   z,   w])
proc vec4b*(x:bool; v:Vec2b; w:bool) : Vec4b {.inline.} = Vec4b(arr: [  x, v.x, v.y,   w])
proc vec4b*(x,y:bool; v:Vec2b)       : Vec4b {.inline.} = Vec4b(arr: [  x,   y, v.x, v.y])
proc vec4b*(x:bool)                  : Vec4b {.inline.} = Vec4b(arr: [  x,   x,   x,   x])

proc vec3b*(x,y,z:bool)      : Vec3b {.inline.} = Vec3b(arr: [  x,   y,   z])
proc vec3b*(v:Vec2b; z:bool) : Vec3b {.inline.} = Vec3b(arr: [v.x, v.y,   z])
proc vec3b*(x:bool; v:Vec2b) : Vec3b {.inline.} = Vec3b(arr: [  x, v.x, v.y])
proc vec3b*(x:bool)          : Vec3b {.inline.} = Vec3b(arr: [  x,   x,   x])

proc vec2b*(x,y:bool) : Vec2b {.inline.} = Vec2b(arr: [x,y])
proc vec2b*(x:bool)   : Vec2b {.inline.} = Vec2b(arr: [x,x])

proc vec4b*(a:array[0..3, bool]) : Vec4b {.inline.} = Vec4b(arr: [a[0], a[1], a[2], a[3]])
proc vec3b*(a:array[0..2, bool]) : Vec3b {.inline.} = Vec3b(arr: [a[0], a[1], a[2]])
proc vec2b*(a:array[0..1, bool]) : Vec2b {.inline.} = Vec2b(arr: [a[0], a[1]])

# conversions
proc  vec4f*[T]  (v : Vec[4, T]) : Vec4f  {.inline.} = Vec4f(  arr: [v.x.float32, v.y.float32, v.z.float32, v.w.float32])
proc  vec3f*[T]  (v : Vec[3, T]) : Vec3f  {.inline.} = Vec3f(  arr: [v.x.float32, v.y.float32, v.z.float32])
proc  vec2f*[T]  (v : Vec[2, T]) : Vec2f  {.inline.} = Vec2f(  arr: [v.x.float32, v.y.float32])
proc  vec4d*[T]  (v : Vec[4, T]) : Vec4d  {.inline.} = Vec4d(  arr: [v.x.float64, v.y.float64, v.z.float64, v.w.float64])
proc  vec3d*[T]  (v : Vec[3, T]) : Vec3d  {.inline.} = Vec3d(  arr: [v.x.float64, v.y.float64, v.z.float64])
proc  vec2d*[T]  (v : Vec[2, T]) : Vec2d  {.inline.} = Vec2d(  arr: [v.x.float64, v.y.float64])
proc  vec4i*[T]  (v : Vec[4, T]) : Vec4i  {.inline.} = Vec4i(  arr: [v.x.int32,   v.y.int32,   v.z.int32,   v.w.int32])
proc  vec3i*[T]  (v : Vec[3, T]) : Vec3i  {.inline.} = Vec3i(  arr: [v.x.int32,   v.y.int32,   v.z.int32])
proc  vec2i*[T]  (v : Vec[2, T]) : Vec2i  {.inline.} = Vec2i(  arr: [v.x.int32,   v.y.int32])
proc  vec4l*[T]  (v : Vec[4, T]) : Vec4l  {.inline.} = Vec4l(  arr: [v.x.int64,   v.y.int64,   v.z.int64,   v.w.int64])
proc  vec3l*[T]  (v : Vec[3, T]) : Vec3l  {.inline.} = Vec3l(  arr: [v.x.int64,   v.y.int64,   v.z.int64])
proc  vec2l*[T]  (v : Vec[2, T]) : Vec2l  {.inline.} = Vec2l(  arr: [v.x.int64,   v.y.int64])
proc  vec4ui*[T] (v : Vec[4, T]) : Vec4ui {.inline.} = Vec4ui( arr: [v.x.uint32,  v.y.uint32,  v.z.uint32,  v.w.uint32])
proc  vec3ui*[T] (v : Vec[3, T]) : Vec3ui {.inline.} = Vec3ui( arr: [v.x.uint32,  v.y.uint32,  v.z.uint32])
proc  vec2ui*[T] (v : Vec[2, T]) : Vec2ui {.inline.} = Vec2ui( arr: [v.x.uint32,  v.y.uint32])
proc  vec4ul*[T] (v : Vec[4, T]) : Vec4ul {.inline.} = Vec4ul( arr: [v.x.uint64,  v.y.uint64,  v.z.uint64,  v.w.uint64])
proc  vec3ul*[T] (v : Vec[3, T]) : Vec3ul {.inline.} = Vec3ul( arr: [v.x.uint64,  v.y.uint64,  v.z.uint64])
proc  vec2ul*[T] (v : Vec[2, T]) : Vec2ul {.inline.} = Vec2ul( arr: [v.x.uint64,  v.y.uint64])
proc  vec4b*[T]  (v : Vec[4, T]) : Vec4b  {.inline.} = Vec4b(  arr: [v.x.bool,    v.y.bool,    v.z.bool,    v.w.bool])
proc  vec3b*[T]  (v : Vec[3, T]) : Vec3b  {.inline.} = Vec3b(  arr: [v.x.bool,    v.y.bool,    v.z.bool])
proc  vec2b*[T]  (v : Vec[2, T]) : Vec2b  {.inline.} = Vec2b(  arr: [v.x.bool,    v.y.bool])


#[
proc vec4f*(v         : Vec4d) : Vec4f      {.inline.} = Vec4f(arr: [v.x.float32, v.y.float32, v.z.float32,   v.w.float32])
proc vec4f*(v         : Vec4i) : Vec4f      {.inline.} = Vec4f(arr: [v.x.float32, v.y.float32, v.z.float32,   v.w.float32])
proc vec4f*(v         : Vec4l) : Vec4f      {.inline.} = Vec4f(arr: [v.x.float32, v.y.float32, v.z.float32,   v.w.float32])
proc vec4f*(v         : Vec4b) : Vec4f      {.inline.} = Vec4f(arr: [v.x.float32, v.y.float32, v.z.float32,   v.w.float32])

proc vec4d*(v: Vec4f) : Vec4d {.inline.} = Vec4d(arr: [v.x.float64, v.y.float64, v.z.float64, v.w.float64])
proc vec4d*(v: Vec4i) : Vec4d {.inline.} = Vec4d(arr: [v.x.float64, v.y.float64, v.z.float64, v.w.float64])
proc vec4d*(v: Vec4l) : Vec4d {.inline.} = Vec4d(arr: [v.x.float64, v.y.float64, v.z.float64, v.w.float64])
proc vec4d*(v: Vec4b) : Vec4d {.inline.} = Vec4d(arr: [v.x.float64, v.y.float64, v.z.float64, v.w.float64])


proc vec4i*(v: Vec4f)  : Vec4i {.inline.} = Vec4i(arr: [v.x.int32, v.y.int32, v.z.int32, v.w.int32])
proc vec4i*(v: Vec4d)  : Vec4i {.inline.} = Vec4i(arr: [v.x.int32, v.y.int32, v.z.int32, v.w.int32])
proc vec4i*(v: Vec4l)  : Vec4i {.inline.} = Vec4i(arr: [v.x.int32, v.y.int32, v.z.int32, v.w.int32])
proc vec4i*(v: Vec4ui) : Vec4i {.inline.} = Vec4i(arr: [v.x.int32, v.y.int32, v.z.int32, v.w.int32])
proc vec4i*(v: Vec4ul) : Vec4i {.inline.} = Vec4i(arr: [v.x.int32, v.y.int32, v.z.int32, v.w.int32])
proc vec4i*(v: Vec4b)  : Vec4i {.inline.} = Vec4i(arr: [v.x.int32, v.y.int32, v.z.int32, v.w.int32])

proc vec4l*(v: Vec4f)  : Vec4l {.inline.} = Vec4l(arr: [v.x.int64, v.y.int64, v.z.int64, v.w.int64])
proc vec4l*(v: Vec4d)  : Vec4l {.inline.} = Vec4l(arr: [v.x.int64, v.y.int64, v.z.int64, v.w.int64])
proc vec4l*(v: Vec4i)  : Vec4l {.inline.} = Vec4l(arr: [v.x.int64, v.y.int64, v.z.int64, v.w.int64])
proc vec4l*(v: Vec4ui) : Vec4l {.inline.} = Vec4l(arr: [v.x.int64, v.y.int64, v.z.int64, v.w.int64])
proc vec4l*(v: Vec4ul) : Vec4l {.inline.} = Vec4l(arr: [v.x.int64, v.y.int64, v.z.int64, v.w.int64])
proc vec4l*(v: Vec4b)  : Vec4l {.inline.} = Vec4l(arr: [v.x.int64, v.y.int64, v.z.int64, v.w.int64])

proc vec4b*(v: Vec4f)  : Vec4b {.inline.} = Vec4b(arr: [v.x.bool, v.y.bool, v.z.bool, v.w.bool])
proc vec4b*(v: Vec4d)  : Vec4b {.inline.} = Vec4b(arr: [v.x.bool, v.y.bool, v.z.bool, v.w.bool])
proc vec4b*(v: Vec4i)  : Vec4b {.inline.} = Vec4b(arr: [v.x.bool, v.y.bool, v.z.bool, v.w.bool])
proc vec4b*(v: Vec4l)  : Vec4b {.inline.} = Vec4b(arr: [v.x.bool, v.y.bool, v.z.bool, v.w.bool])
proc vec4b*(v: Vec4ui) : Vec4b {.inline.} = Vec4b(arr: [v.x.bool, v.y.bool, v.z.bool, v.w.bool])
proc vec4b*(v: Vec4ul) : Vec4b {.inline.} = Vec4b(arr: [v.x.bool, v.y.bool, v.z.bool, v.w.bool])

proc vec3f*(v: Vec3d)  : Vec3f {.inline.} = Vec3f(arr: [v.x.float32, v.y.float32, v.z.float32])
proc vec3f*(v: Vec3i)  : Vec3f {.inline.} = Vec3f(arr: [v.x.float32, v.y.float32, v.z.float32])
proc vec3f*(v: Vec3l)  : Vec3f {.inline.} = Vec3f(arr: [v.x.float32, v.y.float32, v.z.float32])
proc vec3f*(v: Vec3ui) : Vec3f {.inline.} = Vec3f(arr: [v.x.float32, v.y.float32, v.z.float32])
proc vec3f*(v: Vec3ul) : Vec3f {.inline.} = Vec3f(arr: [v.x.float32, v.y.float32, v.z.float32])
proc vec3f*(v: Vec3b)  : Vec3f {.inline.} = Vec3f(arr: [v.x.float32, v.y.float32, v.z.float32])

proc vec3d*(v: Vec3f)  : Vec3d {.inline.} = Vec3d(arr: [v.x.float64, v.y.float64, v.z.float64])
proc vec3d*(v: Vec3i)  : Vec3d {.inline.} = Vec3d(arr: [v.x.float64, v.y.float64, v.z.float64])
proc vec3d*(v: Vec3l)  : Vec3d {.inline.} = Vec3d(arr: [v.x.float64, v.y.float64, v.z.float64])
proc vec3d*(v: Vec3ui) : Vec3d {.inline.} = Vec3d(arr: [v.x.float64, v.y.float64, v.z.float64])
proc vec3d*(v: Vec3ul) : Vec3d {.inline.} = Vec3d(arr: [v.x.float64, v.y.float64, v.z.float64])
proc vec3d*(v: Vec3b)  : Vec3d {.inline.} = Vec3d(arr: [v.x.float64, v.y.float64, v.z.float64])

proc vec3i*(v: Vec3f)  : Vec3i {.inline.} = Vec3i(arr: [v.x.int32, v.y.int32, v.z.int32])
proc vec3i*(v: Vec3d)  : Vec3i {.inline.} = Vec3i(arr: [v.x.int32, v.y.int32, v.z.int32])
proc vec3i*(v: Vec3l)  : Vec3i {.inline.} = Vec3i(arr: [v.x.int32, v.y.int32, v.z.int32])
proc vec3i*(v: Vec3ui) : Vec3i {.inline.} = Vec3i(arr: [v.x.int32, v.y.int32, v.z.int32])
proc vec3i*(v: Vec3ul) : Vec3i {.inline.} = Vec3i(arr: [v.x.int32, v.y.int32, v.z.int32])
proc vec3i*(v: Vec3b)  : Vec3i {.inline.} = Vec3i(arr: [v.x.int32, v.y.int32, v.z.int32])

proc vec3l*(v: Vec3f)  : Vec3l {.inline.} = Vec3l(arr: [v.x.int64, v.y.int64, v.z.int64])
proc vec3l*(v: Vec3d)  : Vec3l {.inline.} = Vec3l(arr: [v.x.int64, v.y.int64, v.z.int64])
proc vec3l*(v: Vec3i)  : Vec3l {.inline.} = Vec3l(arr: [v.x.int64, v.y.int64, v.z.int64])
proc vec3l*(v: Vec3b)  : Vec3l {.inline.} = Vec3l(arr: [v.x.int64, v.y.int64, v.z.int64])

proc vec3b*(v: Vec3f)  : Vec3b {.inline.} = Vec3b(arr: [v.x.bool, v.y.bool, v.z.bool])
proc vec3b*(v: Vec3d)  : Vec3b {.inline.} = Vec3b(arr: [v.x.bool, v.y.bool, v.z.bool])
proc vec3b*(v: Vec3i)  : Vec3b {.inline.} = Vec3b(arr: [v.x.bool, v.y.bool, v.z.bool])
proc vec3b*(v: Vec3l)  : Vec3b {.inline.} = Vec3b(arr: [v.x.bool, v.y.bool, v.z.bool])

proc vec2f*(v: Vec2d)  : Vec2f {.inline.} = Vec2f(arr: [v.x.float32, v.y.float32])
proc vec2f*(v: Vec2i)  : Vec2f {.inline.} = Vec2f(arr: [v.x.float32, v.y.float32])
proc vec2f*(v: Vec2l)  : Vec2f {.inline.} = Vec2f(arr: [v.x.float32, v.y.float32])
proc vec2f*(v: Vec2b)  : Vec2f {.inline.} = Vec2f(arr: [v.x.float32, v.y.float32])

proc vec2d*(v: Vec2f)  : Vec2d {.inline.} = Vec2d(arr: [v.x.float64, v.y.float64])
proc vec2d*(v: Vec2i)  : Vec2d {.inline.} = Vec2d(arr: [v.x.float64, v.y.float64])
proc vec2d*(v: Vec2l)  : Vec2d {.inline.} = Vec2d(arr: [v.x.float64, v.y.float64])
proc vec2d*(v: Vec2b)  : Vec2d {.inline.} = Vec2d(arr: [v.x.float64, v.y.float64])

proc vec2i*(v: Vec2f)  : Vec2i {.inline.} = Vec2i(arr: [v.x.int32, v.y.int32])
proc vec2i*(v: Vec2d)  : Vec2i {.inline.} = Vec2i(arr: [v.x.int32, v.y.int32])
proc vec2i*(v: Vec2l)  : Vec2i {.inline.} = Vec2i(arr: [v.x.int32, v.y.int32])
proc vec2i*(v: Vec2b)  : Vec2i {.inline.} = Vec2i(arr: [v.x.int32, v.y.int32])

proc vec2l*(v: Vec2f)  : Vec2l {.inline.} = Vec2l(arr: [v.x.int64, v.y.int64])
proc vec2l*(v: Vec2d)  : Vec2l {.inline.} = Vec2l(arr: [v.x.int64, v.y.int64])
proc vec2l*(v: Vec2i)  : Vec2l {.inline.} = Vec2l(arr: [v.x.int64, v.y.int64])
proc vec2l*(v: Vec2b)  : Vec2l {.inline.} = Vec2l(arr: [v.x.int64, v.y.int64])

proc vec2b*(v: Vec2f)  : Vec2b {.inline.} = Vec2b(arr: [v.x.bool, v.y.bool])
proc vec2b*(v: Vec2d)  : Vec2b {.inline.} = Vec2b(arr: [v.x.bool, v.y.bool])
proc vec2b*(v: Vec2i)  : Vec2b {.inline.} = Vec2b(arr: [v.x.bool, v.y.bool])
proc vec2b*(v: Vec2l)  : Vec2b {.inline.} = Vec2b(arr: [v.x.bool, v.y.bool])

]#

# bool operations

proc all*[N](v: Vec[N,bool]): bool =
  for b in v.arr:
    if b: return false
  return true

proc any*[N](v: Vec[N,bool]): bool =
  for b in v.arr:
    if b: return true
  return false

template comparisonOpPerComponent(opName, op: untyped): untyped =
  # TODO this is a good place for simd optimization
  proc opName*[N,T](v,u: Vec[N,T]): Vec[N,bool] {.inline.} =
    for i in 0 ..< N:
      result.arr[i] = op(v.arr[i], u.arr[i])


comparisonOpPerComponent(equal, `==`)
comparisonOpPerComponent(greaterThan, `>`)
comparisonOpPerComponent(greaterThanEqual, `>=`)
comparisonOpPerComponent(lessThan, `<`)
comparisonOpPerComponent(lessThanEqual, `<=`)
comparisonOpPerComponent(neg, `not`)
comparisonOpPerComponent(notEqual, `!=`)

# matlab inspired . operators
comparisonOpPerComponent(`.<`, `<` )
comparisonOpPerComponent(`.<=`,`<=`)
comparisonOpPerComponent(`.==`,`==`)
comparisonOpPerComponent(`.>=`,`>=`)
comparisonOpPerComponent(`.>`, `>` )
comparisonOpPerComponent(`.!=`,`!=`)

when isMainModule:
  var v0 = vec3(1.0, 0.5, 0)
  var u0 = vec3(1.0, 1.0, 0)
  var c = cross(v0,u0)

  var v1 = vec4(1,2,3,4) div 2

  v1.yz += vec2(10)

  v1.zw /= vec2(3)

  echo v1

  for row in columnFormat( vec4(0.001, 1.000, 100.0, 0) ):
    echo row

  for row in columnFormat( vec4(1,10,100,1000) ):
    echo row

  for row in columnFormat( vec4("a", "ab", "abc", "abcd") ):
    echo row

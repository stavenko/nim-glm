#import macros.vector
#import macros.functions
#import macros

import strutils
import sequtils
import macros,math

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
    for i in 0 ..< N:
      result.arr[i] = op(v.arr[i], u.arr[i])

  proc op*[N,T](v: Vec[N,T]; val: T): Vec[N,T] {.inline.} =
    for i in 0 ..< N:
      result.arr[i] = op(v.arr[i], val)

  proc op*[N,T](val: T; v: Vec[N,T]): Vec[N,T] {.inline.} =
    for i in 0 ..< N:
      result.arr[i] = op(val, v.arr[i])
    
mathPerComponent(`+`)
mathPerComponent(`-`)
mathPerComponent(`/`)
mathPerComponent(`*`)
mathPerComponent(`div`)

template mathInpl(opName): untyped =
  proc opName*[N,T](v: var Vec[N,T]; u: Vec[N,T]): void =
    for i in 0 ..< N:
      opName(v.arr[i], u.arr[i])

  proc opName*[N,T](v: var Vec[N,T]; x: T): void =
    for i in 0 ..< N:
      opName(v.arr[i], x)
  
mathInpl(`+=`)
mathInpl(`-=`)
mathInpl(`*=`)
mathInpl(`/=`)

proc `[]=`*[N,T](v:var Vec[N,T]; ix:int; c:T): void {.inline.} =
    v.arr[ix] = c
proc `[]`*[N,T](v: Vec[N,T]; ix: int): T {.inline.} =
  v.arr[ix]
proc `[]`*[N,T](v: var Vec[N,T]; ix: int): var T {.inline.} =
  v.arr[ix]

#########################
# constructor functions #
#########################

proc vec2*[T](x,y: T): Vec[2,T] {.inline.} =
  result.arr = [x,y]

proc vec2*[T](arg: T): Vec[2,T] {.inline.} =
  result.arr = [arg,arg]
  
proc vec3*[T](x,y,z : T): Vec[3,T] {.inline.} =
  result.arr = [x,y,z]

proc vec3*[T](arg : T): Vec[3,T] {.inline.} =
  result.arr = [arg,arg,arg]

proc vec4*[T](x,y,z,w : T): Vec[4,T] {.inline.} =
  result.arr = [x,y,z,w]

proc vec4*[T](arg : T): Vec[4,T] {.inline.} =
  result.arr = [arg,arg,arg,arg]
  

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
    

  else:
    let lit = newLit(indices[0])
    result.add head quote do:
      proc `getIdent`*[N,T](v: Vec[N,T]): T {.inline.} =
        v.arr[`lit`]

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
      proc `setIdent`*[N,T](`v1`: Vec[N,T]; `v2`: Vec[`N2lit`,T]): void =
        `assignments`
    
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

proc dot*[N,T](u,v: Vec[N,T]): T {. inline .} =
  # TODO this really should have some simd optimization
  # matrix multiplication is based on this
  for i in 0 ..< N:
    result += u[i] * v[i]
    
proc caddr*[N,T](v:var Vec[N,T]): ptr T {.inline.}=
  ## Address getter to pass vector to native-C openGL functions as pointers
  v.arr[0].addr

proc length2*(v: Vec): auto = dot(v,v)
proc length*(v: Vec): auto = sqrt(dot(v,v))

proc cross*[T](v1,v2:Vec[3,T]): Vec[3,T] =
  v1.yzx * v2.zxy - v1.zxy * v2.yzx

proc normalize*[N,T](v: Vec[N,T]): Vec[N,T] = v * (T(1) / v.length)

proc floor*[N,T](v : Vec[N,T]) : Vec[N,T] =
  for i in 0 .. N:
    result.arr[i] = floor(v.arr[i])

proc ceil*[N,T](v: Vec[N,T]): Vec[N,T] =
  for i in 0 .. N:
    result.arr[i] = ceil(v.arr[i])
  
proc clamp*[N,T](arg: Vec[N,T]; minVal, maxVal: T): Vec[N,T] =
  for i in 0 .. N:
    result.arr[i] = clamp(arg.arr[i], minVal, maxVal)

proc clamp*[N,T](arg, minVal, maxVal: Vec[N,T]): Vec[N,T] =
  for i in 0 .. N:
    result.arr[i] = clamp(arg.arr[i], minVal.arr[i], maxVal.arr[i])
  

##############
# type names #
##############

type
  Vec4*[T] = Vec[4,T]
  Vec3*[T] = Vec[3,T]
  Vec2*[T] = Vec[2,T]

#proc vec4*[T](x,y,z,w:T)         : Vec4[T] = Vec4[T](arr: [  x,   y,   z,   w])
proc vec4*[T](v:Vec3[T],w:T)     : Vec4[T] = Vec4[T](arr: [v.x, v.y, v.z,   w])
proc vec4*[T](x:T,v:Vec3[T])     : Vec4[T] = Vec4[T](arr: [  x, v.x, v.y, v.z])
proc vec4*[T](a,b:Vec2[T])       : Vec4[T] = Vec4[T](arr: [a.x, a.y, b.x, b.y])
proc vec4*[T](v:Vec2[T],z,w:T)   : Vec4[T] = Vec4[T](arr: [v.x, v.y,   z,   w])
proc vec4*[T](x:T,v:Vec2[T],w:T) : Vec4[T] = Vec4[T](arr: [  x, v.x, v.y,   w])
proc vec4*[T](x,y:T,v:Vec2[T])   : Vec4[T] = Vec4[T](arr: [  x,   y, v.x, v.y])
#proc vec4*[T](x:T)               : Vec4[T] = Vec4[T](arr: [  x,   x,   x,   x])

#proc vec3*[T](x,y,z: T)      : Vec3[T] = Vec3[T](arr: [  x,   y,   z])
proc vec3*[T](v:Vec2[T],z:T) : Vec3[T] = Vec3[T](arr: [v.x, v.y,   z])
proc vec3*[T](x:T,v:Vec2[T]) : Vec3[T] = Vec3[T](arr: [  x, v.x, v.y])
#proc vec3*[T](x:T)           : Vec3[T] = Vec3[T](arr: [  x,   x,   x])

#proc vec2*[T](x,y:T) : Vec2[T] = Vec2[T](arr: [x,y])
#proc vec2*[T](x:T)   : Vec2[T] = Vec2[T](arr: [x,x])

  
type
  Vec4u8* = Vec[4, uint8]
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

proc vec4f*(x,y,z,w:float32)             : Vec4f = Vec4f(arr: [  x,   y,   z,   w])
proc vec4f*(v:Vec3f,w:float32)           : Vec4f = Vec4f(arr: [v.x, v.y, v.z,   w])
proc vec4f*(x:float32,v:Vec3f)           : Vec4f = Vec4f(arr: [  x, v.x, v.y, v.z])
proc vec4f*(a,b:Vec2f)                   : Vec4f = Vec4f(arr: [a.x, a.y, b.x, b.y])
proc vec4f*(v:Vec2f,z,w:float32)         : Vec4f = Vec4f(arr: [v.x, v.y,   z,   w])
proc vec4f*(x:float32,v:Vec2f,w:float32) : Vec4f = Vec4f(arr: [  x, v.x, v.y,   w])
proc vec4f*(x,y:float32,v:Vec2f)         : Vec4f = Vec4f(arr: [  x,   y, v.x, v.y])
proc vec4f*(x:float32)                   : Vec4f = Vec4f(arr: [  x,   x,   x,   x])

proc vec3f*(x,y,z:   float32)  : Vec3f = Vec3f(arr: [  x,   y,   z])
proc vec3f*(v:Vec2f,z:float32) : Vec3f = Vec3f(arr: [v.x, v.y,   z])
proc vec3f*(x:float32,v:Vec2f) : Vec3f = Vec3f(arr: [  x, v.x, v.y])
proc vec3f*(x:float32)         : Vec3f = Vec3f(arr: [  x,   x,   x])

proc vec2f*(x,y:float32) : Vec2f = Vec2f(arr: [x,y])
proc vec2f*(x:float32)   : Vec2f = Vec2f(arr: [x,x])

proc vec4f*(a:array[0..3, float32]) : Vec4f = Vec4f(arr: [a[0], a[1], a[2], a[3]])
proc vec3f*(a:array[0..2, float32]) : Vec3f = Vec3f(arr: [a[0], a[1], a[2]])
proc vec2f*(a:array[0..1, float32]) : Vec2f = Vec2f(arr: [a[0], a[1]])

proc vec4d*(x,y,z,w:float64)             : Vec4d = Vec4d(arr: [  x,   y,   z,   w])
proc vec4d*(v:Vec3d,w:float64)           : Vec4d = Vec4d(arr: [v.x, v.y, v.z,   w])
proc vec4d*(x:float64,v:Vec3d)           : Vec4d = Vec4d(arr: [  x, v.x, v.y, v.z])
proc vec4d*(a,b:Vec2d)                   : Vec4d = Vec4d(arr: [a.x, a.y, b.x, b.y])
proc vec4d*(v:Vec2d,z,w:float64)         : Vec4d = Vec4d(arr: [v.x, v.y,   z,   w])
proc vec4d*(x:float64,v:Vec2d,w:float64) : Vec4d = Vec4d(arr: [  x, v.x, v.y,   w])
proc vec4d*(x,y:float64,v:Vec2d)         : Vec4d = Vec4d(arr: [  x,   y, v.x, v.y])
proc vec4d*(x:float64)                   : Vec4d = Vec4d(arr: [  x,   x,   x,   x])

proc vec3d*(x,y,z:   float64)  : Vec3d = Vec3d(arr: [  x,   y,   z])
proc vec3d*(v:Vec2d,z:float64) : Vec3d = Vec3d(arr: [v.x, v.y,   z])
proc vec3d*(x:float64,v:Vec2d) : Vec3d = Vec3d(arr: [  x, v.x, v.y])
proc vec3d*(x:float64)         : Vec3d = Vec3d(arr: [  x,   x,   x])

proc vec2d*(x,y:float64) : Vec2d = Vec2d(arr: [x,y])
proc vec2d*(x:float64)   : Vec2d = Vec2d(arr: [x,x])


proc vec4i*(x,y,z,w:int32)             : Vec4i = Vec4i(arr: [  x,   y,   z,   w])
proc vec4i*(v:Vec3i; w:int32)          : Vec4i = Vec4i(arr: [v.x, v.y, v.z,   w])
proc vec4i*(x:int32; v:Vec3i)          : Vec4i = Vec4i(arr: [  x, v.x, v.y, v.z])
proc vec4i*(a,b:Vec2i)                 : Vec4i = Vec4i(arr: [a.x, a.y, b.x, b.y])
proc vec4i*(v:Vec2i; z,w:int32)        : Vec4i = Vec4i(arr: [v.x, v.y,   z,   w])
proc vec4i*(x:int32; v:Vec2i; w:int32) : Vec4i = Vec4i(arr: [  x, v.x, v.y,   w])
proc vec4i*(x,y:int32; v:Vec2i)        : Vec4i = Vec4i(arr: [  x,   y, v.x, v.y])
proc vec4i*(x:int32)                   : Vec4i = Vec4i(arr: [  x,   x,   x,   x])

proc vec3i*(x,y,z:int32)      : Vec3i = Vec3i(arr: [  x,   y,   z])
proc vec3i*(v:Vec2i; z:int32) : Vec3i = Vec3i(arr: [v.x, v.y,   z])
proc vec3i*(x:int32; v:Vec2i) : Vec3i = Vec3i(arr: [  x, v.x, v.y])
proc vec3i*(x:int32)          : Vec3i = Vec3i(arr: [  x,   x,   x])

proc vec2i*(x,y:int32) : Vec2i = Vec2i(arr: [x,y])
proc vec2i*(x:int32)   : Vec2i = Vec2i(arr: [x,x])

proc vec4i*(a:array[0..3, int32]) : Vec4i = Vec4i(arr: [a[0], a[1], a[2], a[3]])
proc vec3i*(a:array[0..2, int32]) : Vec3i = Vec3i(arr: [a[0], a[1], a[2]])
proc vec2i*(a:array[0..1, int32]) : Vec2i = Vec2i(arr: [a[0], a[1]])

# conversions

proc vec4f*(v: Vec4d) : Vec4f {.inline.} = Vec4f(arr: [v.x.float32, v.y.float32, v.z.float32, v.w.float32])
proc vec4f*(v: Vec4i) : Vec4f {.inline.} = Vec4f(arr: [v.x.float32, v.y.float32, v.z.float32, v.w.float32])
proc vec4f*(v: Vec4l) : Vec4f {.inline.} = Vec4f(arr: [v.x.float32, v.y.float32, v.z.float32, v.w.float32])
proc vec4d*(v: Vec4f) : Vec4d {.inline.} = Vec4d(arr: [v.x.float64, v.y.float64, v.z.float64, v.w.float64])
proc vec4d*(v: Vec4i) : Vec4d {.inline.} = Vec4d(arr: [v.x.float64, v.y.float64, v.z.float64, v.w.float64])
proc vec4d*(v: Vec4l) : Vec4d {.inline.} = Vec4d(arr: [v.x.float64, v.y.float64, v.z.float64, v.w.float64])
proc vec4i*(v: Vec4f) : Vec4i {.inline.} = Vec4i(arr: [v.x.int32, v.y.int32, v.z.int32, v.w.int32])
proc vec4i*(v: Vec4i) : Vec4i {.inline.} = Vec4i(arr: [v.x.int32, v.y.int32, v.z.int32, v.w.int32])
proc vec4i*(v: Vec4l) : Vec4i {.inline.} = Vec4i(arr: [v.x.int32, v.y.int32, v.z.int32, v.w.int32])
proc vec4l*(v: Vec4f) : Vec4l {.inline.} = Vec4l(arr: [v.x.int64, v.y.int64, v.z.int64, v.w.int64])
proc vec4l*(v: Vec4i) : Vec4l {.inline.} = Vec4l(arr: [v.x.int64, v.y.int64, v.z.int64, v.w.int64])
proc vec4l*(v: Vec4l) : Vec4l {.inline.} = Vec4l(arr: [v.x.int64, v.y.int64, v.z.int64, v.w.int64])
proc vec3f*(v: Vec3d) : Vec3f {.inline.} = Vec3f(arr: [v.x.float32, v.y.float32, v.z.float32])
proc vec3f*(v: Vec3i) : Vec3f {.inline.} = Vec3f(arr: [v.x.float32, v.y.float32, v.z.float32])
proc vec3f*(v: Vec3l) : Vec3f {.inline.} = Vec3f(arr: [v.x.float32, v.y.float32, v.z.float32])
proc vec3d*(v: Vec3f) : Vec3d {.inline.} = Vec3d(arr: [v.x.float64, v.y.float64, v.z.float64])
proc vec3d*(v: Vec3i) : Vec3d {.inline.} = Vec3d(arr: [v.x.float64, v.y.float64, v.z.float64])
proc vec3d*(v: Vec3l) : Vec3d {.inline.} = Vec3d(arr: [v.x.float64, v.y.float64, v.z.float64])
proc vec3i*(v: Vec3f) : Vec3i {.inline.} = Vec3i(arr: [v.x.int32, v.y.int32, v.z.int32])
proc vec3i*(v: Vec3i) : Vec3i {.inline.} = Vec3i(arr: [v.x.int32, v.y.int32, v.z.int32])
proc vec3i*(v: Vec3l) : Vec3i {.inline.} = Vec3i(arr: [v.x.int32, v.y.int32, v.z.int32])
proc vec3l*(v: Vec3f) : Vec3l {.inline.} = Vec3l(arr: [v.x.int64, v.y.int64, v.z.int64])
proc vec3l*(v: Vec3i) : Vec3l {.inline.} = Vec3l(arr: [v.x.int64, v.y.int64, v.z.int64])
proc vec3l*(v: Vec3l) : Vec3l {.inline.} = Vec3l(arr: [v.x.int64, v.y.int64, v.z.int64])
proc vec2f*(v: Vec2d) : Vec2f {.inline.} = Vec2f(arr: [v.x.float32, v.y.float32])
proc vec2f*(v: Vec2i) : Vec2f {.inline.} = Vec2f(arr: [v.x.float32, v.y.float32])
proc vec2f*(v: Vec2l) : Vec2f {.inline.} = Vec2f(arr: [v.x.float32, v.y.float32])
proc vec2d*(v: Vec2f) : Vec2d {.inline.} = Vec2d(arr: [v.x.float64, v.y.float64])
proc vec2d*(v: Vec2i) : Vec2d {.inline.} = Vec2d(arr: [v.x.float64, v.y.float64])
proc vec2d*(v: Vec2l) : Vec2d {.inline.} = Vec2d(arr: [v.x.float64, v.y.float64])
proc vec2i*(v: Vec2f) : Vec2i {.inline.} = Vec2i(arr: [v.x.int32, v.y.int32])
proc vec2i*(v: Vec2i) : Vec2i {.inline.} = Vec2i(arr: [v.x.int32, v.y.int32])
proc vec2i*(v: Vec2l) : Vec2i {.inline.} = Vec2i(arr: [v.x.int32, v.y.int32])
proc vec2l*(v: Vec2f) : Vec2l {.inline.} = Vec2l(arr: [v.x.int64, v.y.int64])
proc vec2l*(v: Vec2i) : Vec2l {.inline.} = Vec2l(arr: [v.x.int64, v.y.int64])
proc vec2l*(v: Vec2l) : Vec2l {.inline.} = Vec2l(arr: [v.x.int64, v.y.int64])
  
if isMainModule:
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


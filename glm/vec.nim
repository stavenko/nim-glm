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


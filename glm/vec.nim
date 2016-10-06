import macros.vector
import macros.functions
import macros
import strutils
import sequtils
import math

##Vector module contains all types and functions to manipulate vectors
##
type
  Vec*[N : static[int], T] = object
    arr: array[N, T]
    
proc `$`*(v: Vec) : string =  $v.arr

template mathPerComponent(op: untyped): untyped =
  proc op*(v,u: Vec): Vec =
    for i in 0 ..< Vec.N:
      result[i] = op(v[i], u[i])

  proc op*[N,T](v: Vec[N,T]; val: T): Vec[N,T] =
    for i in 0 ..< N:
      result[i] = op(v[i], val)

  proc op*[N,T](val: T; v: Vec[N,T]): Vec[N,T] =
    for i in 0 ..< N:
      result[i] = op(val, v[i])


    
mathPerComponent(`+`)
mathPerComponent(`-`)
mathPerComponent(`/`)
mathPerComponent(`*`)

template mathInpl(op: untyped): untyped =
  proc op*[N,T](v: var Vec[N,T]; u: Vec[N,T]): void =
    for i in 0 ..< N:
      op(v.arr[i], u[i])

mathInpl(`+=`)
mathInpl(`-=`)
mathInpl(`*=`)
mathInpl(`/=`)

proc `[]=`*[N,T](v:var Vec[N,T]; ix:int; c:T) = v.arr[ix] = c
proc `[]`*[N,T](v: Vec[N,T]; ix: int): T = v.arr[ix]
proc `[]`*[N,T](v: var Vec[N,T]; ix: int): var T = v.arr[ix]

proc vec2*[T](x,y: T): Vec[2,T] =
  result[0] = x
  result[1] = y

proc vec2*[T](arg: T): Vec[2,T] =
  result[0] = arg
  result[1] = arg
  
proc vec3*[T](x,y,z : T): Vec[3,T] =
  result[0] = x
  result[1] = y
  result[2] = z

proc vec3*[T](arg : T): Vec[3,T] =
  result[0] = arg
  result[1] = arg
  result[2] = arg

proc vec4*[T](x,y,z,w : T): Vec[4,T] =
  result[0] = x
  result[1] = y
  result[2] = z
  result[3] = w

proc vec4*[T](arg : T): Vec[4,T] =
  result[0] = arg
  result[1] = arg
  result[2] = arg
  result[3] = arg
  
proc subVec[N,T](v: var Vec[N,T]; offset, length: static[int]): var Vec[length,T] =
  cast[ptr Vec[length, T]](v.arr[offset].addr)[]
  
proc head(node: NimNode): NimNode {.compileTime.} = node[0]
  
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

  
proc swizzleMethods(indices: varargs[int]) : seq[NimNode] {.compileTime.}=
  result.newSeq(0)
  
  const chars = "xyzw"

  var name = ""
  for idx in indices:
    name.add chars[idx]

  let getIdent = ident(name)
  let setIdent = ident(name & '=')

  if indices.len > 1:
    let constructCall = newCall(ident("vec" & $indices.len))
    let Nlit = newLit(indices.len)
    let v = genSym(nskParam, "v")

    for idx in indices:
      let lit = newLit(idx)
      constructCall.add head quote do:
        `v`[`lit`]

    result.add head quote do:
      proc `getIdent`*[N,T](`v`: Vec[N,T]): Vec[`Nlit`,T] = `constructCall`

    if continuousIndices(indices):
      let offsetLit = newLit(indices[0])
      let lengthLit = newLit(indices.len)
      result.add head quote do:
        proc `getIdent`*[N,T](`v`: var Vec[N,T]): var Vec[`Nlit`,T] =
          `v`.subVec(`offsetLit`, `lengthLit`)

  else:
    let lit = newLit(indices[0])
    result.add head quote do:
      proc `getIdent`*(v: Vec): auto = v[`lit`]

  if growingIndices(indices):
    let N2lit = newLit(indices.len)
    let v1 = genSym(nskParam, "v1")
    let v2 = genSym(nskParam, "v2")

    let assignments = newStmtList()
    for i,idx in indices:
      let litL = newLit(idx)
      let litR = newLit(i)
      assignments.add head quote do:
        `v1`[`litL`] = `v2`[`litR`]
      
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

proc dot*[N,T](u,v: Vec[N,T]): T =
  for i in 0 ..< N:
    result += u[i] * v[i]
    
proc caddr*[N,T](v:var Vec[N,T]): ptr T =
  ## Address getter to pass vector to native-C openGL functions as pointers
  v.arr[0].addr

proc length2*(v: Vec): auto = dot(v,v)
proc length*(v: Vec): auto = sqrt(dot(v,v))

proc cross*[T](x,y:Vec[3,T]): Vec[3,T] =
    vec3(x.y * y.z - y.y * x.z,
         x.z * y.x - y.z * x.x,
         x.x * y.y - y.x * x.y)

#normalizeMacros(MAX_VEC_SIZE)

if isMainModule:
    var v0 = vec3(1.0, 0.5, 0)
    var u0 = vec3(1.0, 1.0, 0)
    var c = cross(v0,u0)

    var v1 = vec4(1,2,3,4)

    v1.yz += vec2(10)

    echo v1


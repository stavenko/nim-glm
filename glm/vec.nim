import macros.vector
import macros.functions
import macros
import strutils
import sequtils
import math

##Vector module contains all types and functions to manipulate vectors
##
type Vec*[N : static[int], T] = distinct array[N, T]
proc `$`*(v: Vec) : string =  $ array[v.N, v.T](v)


template mathPerComponent(op: untyped): untyped =
  proc op*(v,u: Vec): Vec =
    for i in 0 ..< Vec.N:
      result[i] = op(v[i], u[0])

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

proc `[]=`*[N,T](v:var Vec[N,T]; ix:int; c:T) = array[N,T](v)[ix] = c
proc `[]`*[N,T](v: Vec[N,T]; ix: int): T = array[N,T](v)[ix]

proc vec2*[T](x,y: T): Vec[2,T] =
  result[0] = x
  result[1] = y

proc vec3*[T](x,y,z : T): Vec[3,T] =
  result[0] = x
  result[1] = y
  result[2] = z
  
proc vec4*[T](x,y,z,w : T): Vec[4,T] =
  result[0] = x
  result[1] = y
  result[2] = z
  result[3] = w

macro genSwizzleOps*(): untyped =
  result = newStmtList()

  const chars = "xyzw"

  proc name(args: varargs[int]) : string =
    result = ""
    for arg in args:
      result.add chars[arg]

  for i in 0 .. chars.high:
    let getIdent = ident(name(i))
    let setIdent = ident(name(i)&'=')
    let iLit = newLit(i)
    
    result.add quote do:
      proc `getIdent`*(v: Vec): auto = v[`iLit`]
      proc `setIdent`*[N,T](v: Vec[N,T]; val: T): void =
        v[`iLit`] = val
      
    for j in 0 .. chars.high:
      let getIdent = ident(name(i,j))
      let setIdent = ident(name(i,j)&'=')
      let jLit = newLit(j)
        
      result.add quote do:
        proc `getIdent`*(v: Vec): auto = vec2(v[`iLit`], v[`jLit`])

      if i < j:
        result.add quote do:
          proc `setIdent`*[N,T](v: Vec[N,T]; vi, vj: T): void =
            v[`iLit`] = vi
            v[`jLit`] = vj

      for k in 0 .. chars.high:
        let getIdent = ident(name(i,j,k))
        let setIdent = ident(name(i,j,k)&'=')
        let kLit = newLit(k)
        
        result.add quote do:
          proc `getIdent`*(v: Vec): auto = vec3(v[`iLit`], v[`jLit`], v[`kLit`])

        if i < j and j < k:
          result.add quote do:
            proc `setIdent`*[N,T](v: Vec[N,T]; vi, vj, vk: T): void =
              v[`iLit`] = vi
              v[`jLit`] = vj
              v[`kLit`] = vk

        for m in 0 .. chars.high:
          let getIdent = ident(name(i,j,k,m))
          let setIdent = ident(name(i,j,k,m)&'=')
          let mLit = newLit(m)
        
          result.add quote do:
            proc `getIdent`*(v: Vec): auto = vec4(v[`iLit`], v[`jLit`], v[`kLit`], v[`mLit`])
          if i < j and j < k and k < m:
            if i < j and j < k:
              result.add quote do:
                proc `setIdent`*[N,T](v: Vec[N,T]; vi, vj, vk, vm: T): void =
                  v[`iLit`] = vi
                  v[`jLit`] = vj
                  v[`kLit`] = vk
                  v[`mLit`] = vm

  echo result.repr
  
genSwizzleOps()

proc dot*[N,T](u,v: Vec[N,T]): T =
  for i in 0 ..< N:
    result += u[i] * v[i]
    
proc caddr*[N,T](v:var Vec[N,T]): ptr T =
  ## Address getter to pass vector to native-C openGL functions as pointers
  array[N, T](v)[0].addr

proc length2*(v: Vec): auto = dot(v,v)
proc length*(v: Vec): auto = sqrt(dot(v,v))

proc cross*[T](x,y:Vec[3,T]): Vec[3,T] =
    vec3(x.y * y.z - y.y * x.z,
         x.z * y.x - y.z * x.x,
         x.x * y.y - y.x * x.y)

#normalizeMacros(MAX_VEC_SIZE)

if isMainModule:
    var v = vec3(1.0, 0.5, 0)
    var u = vec3(1.0, 1.0, 0)
    var c = cross(v,u)
    echo "Should not use this as main module: $# [$#, $#]" % [$c , $dot(c,v), $dot(c,u)]




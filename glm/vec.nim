import macros.vector
import macros.functions
import strutils
import sequtils
import math 

static:
    const MAX_VEC_SIZE*:int = 4

##Vector module contains all types and functions to manipulate vectors
##
defineVectorTypes(MAX_VEC_SIZE)
mkToStr(MAX_VEC_SIZE)
mkMathPerComponent(MAX_VEC_SIZE)
arrGetters(MAX_VEC_SIZE)
arrSetters(MAX_VEC_SIZE)
componentGetterSetters( MAX_VEC_SIZE )
multiComponentGetterList( MAX_VEC_SIZE )
createConstructors( MAX_VEC_SIZE )
addrGetter(MAX_VEC_SIZE)
createScalarOperations(MAX_VEC_SIZE)
createDotProduct(MAX_VEC_SIZE)
createLengths(MAX_VEC_SIZE)

proc cross*[T](x,y:Vec3[T]):Vec3[T]=
    vec3(x.y * y.z - y.y * x.z,
         x.z * y.x - y.z * x.x,
         x.x * y.y - y.x * x.y)

normalizeMacros(MAX_VEC_SIZE)

if isMainModule:
    var v = vec3(1.0, 0.5, 0)
    var u = vec3(1.0, 1.0, 0)
    var c = cross(v,u)
    echo "Should not use this as main module: $# [$#, $#]" % [$c , $dot(c,v), $dot(c,u)]
    var arr = [1,2,3]



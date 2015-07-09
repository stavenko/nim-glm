import vec_definitions
import strutils
static:
    const MAX_VEC_SIZE*:int = 4



defineVectorTypes(MAX_VEC_SIZE)
mkToStr(MAX_VEC_SIZE)
mkMathPerComponent(MAX_VEC_SIZE)
arrGetters(MAX_VEC_SIZE)
arrSetters(MAX_VEC_SIZE)
componentGetterSetters( MAX_VEC_SIZE )
multiComponentGetterList( MAX_VEC_SIZE )
createConstructors( MAX_VEC_SIZE )
    
if isMainModule:
    var v = vec4()
    echo "Should not use this as main module", v


import strutils, macros
import glm.vec

static:
    const
        MIN_MATRIX_SIZE:int = 2
        MAX_MATRIX_SIZE:int = MAX_VEC_SIZE
acro defineMatrixTypes():stmt=
    result = newNimNode(nnkStmtList);
    for col in MIN_MATRIX_SIZE .. MAX_MATRIX_SIZE:
        for row in MIN_MATRIX_SIZE .. MAX_MATRIX_SIZE:
            var def = "type Mat$1x$2[T] = distinct array[$1, Vec$2[T]]" % [$col, $row]
            result.add(parseStmt(def))


defineMatrixTypes()





proc `$`*[T](m:Mat4x4[T]):string = $ array[4, array[4,T]](m)
proc addr[T](m:var Mat4x4[T]):ptr T = array[4, array[4,T]](m)[0][0].addr
proc `[]=`[T](m:var Mat4x4[T], ix:int, c: Vec4[T])= array[4, Vec4[T]](m)[ix]=c
proc `[]`[T](m:var Mat4x4[T], ix:int):var Vec4[T]= array[4, Vec4[T]](m)[ix]


var
    c:float32=0.0
    col = vec4(5.0.float32)
    m:Mat4x4[float32] = Mat4x4([col, col, col, col])

for i in 0..3:
    c += 1.0;
    m[i] = vec4(c)


echo "mat:", m
var mPtr = m.addr
var imPtr:int =cast[int](mPtr)
var sz = sizeof(float32)

echo "ptr:", (m[0]).repr

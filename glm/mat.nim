import strutils, macros
import arrayUtils
import mat_definitions
import vec

static:
    const
        MIN_MATRIX_SIZE:int = 2
        MAX_MATRIX_SIZE:int = MAX_VEC_SIZE

defineMatrixTypes(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE)
matrixEchos(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE)
columnGetters(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE)
columnSetters(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE)
addrGetter(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE)
matrixConstructors(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE)
emptyConstructors(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE)
matrixScalarOperations(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE)

proc inverse*[T](m:var Mat2x2[T]):Mat2x2[T]=
    var
        OneOverDeterminant = (1.T) /
            ( + m[0][0] * m[1][1] -
                m[1][0] * m[0][1])
    result=mat2(vec2( + m[1][1] * OneOverDeterminant,
                      - m[0][1] * OneOverDeterminant),
                vec2( - m[1][0] * OneOverDeterminant,
                      + m[0][0] * OneOverDeterminant))

proc inverse*[T](m:var Mat3x3[T]):Mat3x3[T]=
    var 
        oD = (1.T) /
            ( + m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
                m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2]) +
                m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2])
            )
    result = mat3();
    result[0][0] = + (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * oD
    result[1][0] = - (m[1][0] * m[2][2] - m[2][0] * m[1][2]) * oD;
    result[2][0] = + (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * oD;
    result[0][1] = - (m[0][1] * m[2][2] - m[2][1] * m[0][2]) * oD;
    result[1][1] = + (m[0][0] * m[2][2] - m[2][0] * m[0][2]) * oD;
    result[2][1] = - (m[0][0] * m[2][1] - m[2][0] * m[0][1]) * oD;
    result[0][2] = + (m[0][1] * m[1][2] - m[1][1] * m[0][2]) * oD;
    result[1][2] = - (m[0][0] * m[1][2] - m[1][0] * m[0][2]) * oD;
    result[2][2] = + (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * oD;

proc inverse*[T](m:var Mat4x4[T]):Mat4x4[T]=
    var 
        Coef00:T = (m[2][2]  * m[3][3]) - (m[3][2]  *  m[2][3])
        Coef02:T = (m[1][2]  * m[3][3]) - (m[3][2]  *  m[1][3])
        Coef03:T = (m[1][2]  * m[2][3]) - (m[2][2]  *  m[1][3])

        Coef04:T = (m[2][1]  * m[3][3]) - (m[3][1]  *  m[2][3])
        Coef06:T = (m[1][1]  * m[3][3]) - (m[3][1]  *  m[1][3])
        Coef07:T = (m[1][1]  * m[2][3]) - (m[2][1]  *  m[1][3])

        Coef08:T = (m[2][1]  * m[3][2]) - (m[3][1]  *  m[2][2])
        Coef10:T = (m[1][1]  * m[3][2]) - (m[3][1]  *  m[1][2])
        Coef11:T = (m[1][1]  * m[2][2]) - (m[2][1]  *  m[1][2])

        Coef12:T = (m[2][0]  * m[3][3]) - (m[3][0]  *  m[2][3])
        Coef14:T = (m[1][0]  * m[3][3]) - (m[3][0]  *  m[1][3])
        Coef15:T = (m[1][0]  * m[2][3]) - (m[2][0]  *  m[1][3])

        Coef16:T = (m[2][0]  * m[3][2]) - (m[3][0]  *  m[2][2])
        Coef18:T = (m[1][0]  * m[3][2]) - (m[3][0]  *  m[1][2])
        Coef19:T = (m[1][0]  * m[2][2]) - (m[2][0]  *  m[1][2])

        Coef20:T = (m[2][0]  * m[3][1]) - (m[3][0]  *  m[2][1])
        Coef22:T = (m[1][0]  * m[3][1]) - (m[3][0]  *  m[1][1])
        Coef23:T = (m[1][0]  * m[2][1]) - (m[2][0]  *  m[1][1])

    var
        Fac0 = vec4(Coef00, Coef00, Coef02, Coef03)
        Fac1 = vec4(Coef04, Coef04, Coef06, Coef07)
        Fac2 = vec4(Coef08, Coef08, Coef10, Coef11)
        Fac3 = vec4(Coef12, Coef12, Coef14, Coef15)
        Fac4 = vec4(Coef16, Coef16, Coef18, Coef19)
        Fac5 = vec4(Coef20, Coef20, Coef22, Coef23)

        Vec0=vec4(m[1][0], m[0][0], m[0][0], m[0][0])
        Vec1=vec4(m[1][1], m[0][1], m[0][1], m[0][1])
        Vec2=vec4(m[1][2], m[0][2], m[0][2], m[0][2])
        Vec3=vec4(m[1][3], m[0][3], m[0][3], m[0][3])

        Inv0=vec4((Vec1 * Fac0) - (Vec2 * Fac1) + (Vec3 * Fac2))
        Inv1=vec4((Vec0 * Fac0) - (Vec2 * Fac3) + (Vec3 * Fac4))
        Inv2=vec4((Vec0 * Fac1) - (Vec1 * Fac3) + (Vec3 * Fac5))
        Inv3=vec4((Vec0 * Fac2) - (Vec1 * Fac4) + (Vec2 * Fac5))

        SignA:Vec4[float] = vec4(+1.float, -1, +1, -1)
        SignB:Vec4[float] = vec4(-1.float, +1, -1, +1)
        Inverse = mat4(Inv0 * SignA, Inv1 * SignB, Inv2 * SignA, Inv3 * SignB)

        Row0 = vec4(Inverse[0][0], Inverse[1][0], Inverse[2][0], Inverse[3][0])

        Dot0 = m[0] * Row0
        Dot1 = (Dot0.x + Dot0.y) + (Dot0.z + Dot0.w)

        OneOverDeterminant = (1.T) / Dot1
    result = Inverse * OneOverDeterminant

matrixMultiplication(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE)

#proc `$`*[T](m:Mat4x4[T]):string = $ array[4, array[4,T]](m)
#proc addr[T](m:var Mat4x4[T]):ptr T = array[4, array[4,T]](m)[0][0].addr
#proc `[]=`[T](m:var Mat4x4[T], ix:int, c: Vec4[T])= array[4, Vec4[T]](m)[ix]=c
#proc `[]`[T](m:var Mat4x4[T], ix:int):var Vec4[T]= array[4, Vec4[T]](m)[ix]


#var
    #c:float32=0.0
    #col = vec4(5.0.float32)
    #m:Mat4x4[float32] = Mat4x4([col, col, col, col])

#for i in 0..3:
    #c += 1.0;
    #m[i] = vec4(c)


#echo "mat:", m
#var mPtr = m.addr
#var imPtr:int =cast[int](mPtr)
#var sz = sizeof(float32)

#echo "ptr:", (m[0]).repr
if isMainModule:
    var m= mat4x4();
    var m2 = mat2(vec2(3.float,1), vec2(1.float,2));
    var v = vec4(5.0);
    echo "im",m2, inverse(m2)
    echo "MMA", repr(m.addr)

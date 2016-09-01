import strutils, macros
import ./arrayUtils
import macros.matrix
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
diagonalConstructors(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE)
matrixScalarOperations(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE)
matrixComparison(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE)

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
    result = mat3()
    result[0][0] = + (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * oD
    result[1][0] = - (m[1][0] * m[2][2] - m[2][0] * m[1][2]) * oD
    result[2][0] = + (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * oD
    result[0][1] = - (m[0][1] * m[2][2] - m[2][1] * m[0][2]) * oD
    result[1][1] = + (m[0][0] * m[2][2] - m[2][0] * m[0][2]) * oD
    result[2][1] = - (m[0][0] * m[2][1] - m[2][0] * m[0][1]) * oD
    result[0][2] = + (m[0][1] * m[1][2] - m[1][1] * m[0][2]) * oD
    result[1][2] = - (m[0][0] * m[1][2] - m[1][0] * m[0][2]) * oD
    result[2][2] = + (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * oD

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

        SignA:Vec4[T] = vec4(+1.T, -1, +1, -1)
        SignB:Vec4[T] = vec4(-1.T, +1, -1, +1)
        Inverse = mat4(Inv0 * SignA, Inv1 * SignB, Inv2 * SignA, Inv3 * SignB)

        Row0 = vec4(Inverse[0][0], Inverse[1][0], Inverse[2][0], Inverse[3][0])

        Dot0 = m[0] * Row0
        Dot1 = (Dot0.x + Dot0.y) + (Dot0.z + Dot0.w)

        OneOverDeterminant = (1.T) / Dot1
    result = Inverse * OneOverDeterminant

fromArray(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE)
matrixMultiplication(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE)
matrixVectorMultiplication(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE)
matrixUnaryScalarOperations(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE)

#proc mat4x4*[T](a:array[4,array[4,T]]):Mat4x4[T]=
    #Mat4x4([Vec4(a[0]), Vec4(a[1]), Vec4(a[2]), Vec4(a[3])])

#proc `*`*[T](a:Mat4x4[T], b:Mat4x4[T]):Mat4x4[T]=
    #matProduct(array[4, array[4,T]](a), array[4,array[4,T]](b)).mat4x4

if isMainModule:

    var m22 = mat3(vec3(1.0, 5, 10), vec3(0.66,1,70), vec3(10.0,2.0,1))
    var v2  = vec3(2.0)
    var m22i = inverse(m22)
    var v2m = v2 * m22
    var v2r = v2m * m22i


    m22 *= 3
    echo m22

    echo mat3(5.0)

import glm
import math

proc radians[T](deg:T):T=
    return deg/180*PI;

proc test1()=
    var 
        v1 = vec1()
        v2 = vec2()
        v3 = vec3(v2, 50.0)
        m  = rotate(mat4(), vec3(1.0, 0, 0), math.PI/4)
    echo v3
    echo v1
    v1.x = 10.0
    echo v1
    v1[0] = 30
    echo v1
    echo m, inverse(m)

proc test2()=
    var 
        translate = 4.0
        view = translate(mat4(1.0), vec3(0.0,0, -translate))
        perspective = perspective(radians(45.0), 4.0 / 3.0, 0.1, 100.0)
        RES =  view * perspective
    echo view
    echo perspective
    echo 1/((4.0 / 3.0)* tan(45/2))
    echo RES

proc test3() =
    let
      normal = vec3(1,2,3).normalize
      offset = vec3(1,2,3)
      angle = 128

    var v1 = vec4(1)
    v1 *= 1.234
    v1 *= mat4(1.0).translate(offset).rotate(normal, angle)
    v1.x += 7
    v1.yz += vec2(1,2)
    v1.w *= 2

if isMainModule:
    test1()
    test2()
    test3()

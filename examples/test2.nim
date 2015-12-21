import glm
import math

proc radians[T](deg:T):T=
    return deg/180*PI;

proc main()=
    var 
        translate = 4.0
        view = translate(mat4(1.0), vec3(0.0,0, -translate))
        perspective = perspective(radians(45.0), 4.0 / 3.0, 0.1, 100.0)
        RES =  view * perspective
    echo view
    echo perspective
    echo 1/((4.0 / 3.0)* tan(45/2))
    echo RES


if isMainModule:
    main()

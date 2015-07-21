import glm
import math

proc main()=
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

if isMainModule:
    main()

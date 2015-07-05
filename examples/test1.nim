import glm

proc main()=
    var 
        v1 = vec1()
        v2 = vec2()
        v3 = vec3(v2, 50.0)
    echo v3
    echo v1
    v1.x = 10.0
    echo v1
    v1[0] = 30
    echo v1

if isMainModule:
    main()

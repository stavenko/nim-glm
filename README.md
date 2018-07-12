#Nim-glm port for matrix-vector algebra with shader like syntax.

Nim-glm has vector constructors:
Here's some examples

    var
        v = vec3(1.0, 5.0, 6.0)
        a = vec3(2.0, 2.0, 5.0)
        v4 = vec4(v, 1.0);
        c = cross(v,a)
        m = rotate(mat4(), 5.0, vec3(1.0, 0.0, 0.0))
        r = v4 * m


Also, this version has basics for common matrices creations:

    var
        eye = vec3(50.0, 50.0, 10.0)
        center = vec3(0.0)
        up = vec3(0.0, 1.0, 0.0)
        viewMatrix = lookAt(eye, center, up)
        projectionMat = perspective(math.PI/2, 1.0, 0.01, 100.0)

    echo viewMatrix * projectionMat

Use it in OpenGL environment:

    var modelView = mat4f(1)
      .rotate(alpha, n.x, n.y, n.z)
      .scale(4,5,6)
      .translate(1,2,3)

    glUniformMatrix4fv(_uniformLocation, 1, false, modelView.caddr)

There is swizzling support:

    var pos1,pos2: Vec4f
    pos1.xyz = pos2.zww
    pos1.yz += pos2.ww
    var texcoord: Vec2f
    echo texcoord.st
    var color: Vec4f
    color.rgb = color.bgr


perlin noise:

    import glm/vec
    import glm/noise

    var line = newStringOfCap(80)
    for y in 0 ..< 20:
      for x in 0 ..< 40:
        let n = perlin(vec2f(float32(x), float32(y)) * 0.1'f32)
        let i = int(floor((n + 1) * 5))
        line.add "  .:+=*%#@"[i]
      echo line
      line.setLen(0)


    # expected output:
    #
    #  =+++:::::+======++++==*%%%%**==++++++++=
    #  ===++::::++====++::++=*%%%%%*===++++:+++
    #  ***=+::.::+====++:::+==*%%%%***==++:::::
    #  ***=+:...:++===++:::++==**%%*****=++:...
    #  **=+::...::+====++:::+++==********=+:.
    #  *==+:....::+=====++:::::++==******=+:.
    #  ==+:..  .::+=****=++:::::::+==*****=+:.
    #  =+:..   .:++=*%%**=++::....:+==**%**=+:.
    #  ++:.   ..:+=*%%%%**=+:..   .:+=**%%**=+:
    #  ++:.   .::+=*%###%*=+::.   .:+==*%%%**=+
    #  =+:..  .:+==*%###%*==+:.   .:+==*%%%%*==
    #  =++:....:+=**%###%**=+:.    .:+=*%%#%%*=
    #  ==+::..:++=*%%####%*=+:.    .:+=*%%#%%**
    #  *==+:::++==*%%####%**=+:.   .:+==*%%%%**
    #  **==++++==***%%##%%%**=+:....::+=*****==
    #  %**======******%%%%%%**=+:....:++=***==+
    #  ***===*****====****%%%*=+::...:++====+++
    #  **===*****==++++==******=+::.::++====++:
    #  *====*****=++:::++==****=+::::+++====++:
    #  ==++==***==+::..::+======++::+++======++

* Changes regarding based to C++glm and glsl

  - the `mod` function is called `modulo` instead. `mod` is already an
    operator in Nim and has it's own meaning that is very different to
    the meaning of the `mod` function in glsl. The name `fmod` is also
    not good, because `fmod` in c++ has also a different meaning.
    Therefore `mod` is simply named `modulo` in nim-glm.  The other
    mod functions all have a behavior towards zero, modulo does not
    have this.

  - swizzle support. Unlike c++, Nim allows pretty well to implement
    swizzling. So it is implemented with least amount of surprise.

  - simd instructions are not implemented.  You could hope that some
    day the C compiler will be smart enough to inject them, but I would
    not bet on it.

  - glm in c++ has a lot more extensions that are not yet ported over
    to the Nim version. They are added when needed. That does not mean
    that this library is lacking any important features.

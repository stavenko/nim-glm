import globals
import mat
import vec

proc translate*[T](m:Mat4[T]; v:Vec3[T]): Mat4x4[T] =
  result = m
  result[3] = (m[0] * v[0]) + (m[1] * v[1]) + (m[2] * v[2] ) + m[3]

proc rotate*[T](m:Mat4x4[T]; axis:Vec3[T]; angle:T): Mat4x4[T] =
    let
        a = angle
        c = cos(a)
        s = sin(a)
    var
        naxis = normalize(axis)
        temp  = T(1 - c) * naxis
        Rotate = mat4[T](0)

    Rotate[0,0] = c + temp[0] * axis[0]
    Rotate[0,1] = 0 + temp[0] * axis[1] + s * axis[2]
    Rotate[0,2] = 0 + temp[0] * axis[2] - s * axis[1]

    Rotate[1,0] = 0 + temp[1] * axis[0] - s * axis[2]
    Rotate[1,1] = c + temp[1] * axis[1]
    Rotate[1,2] = 0 + temp[1] * axis[2] + s * axis[0]

    Rotate[2,0] = 0 + temp[2] * axis[0] + s * axis[1]
    Rotate[2,1] = 0 + temp[2] * axis[1] - s * axis[0]
    Rotate[2,2] = c + temp[2] * axis[2]

    result = mat4[T](0)
    result[0] = m[0] * Rotate[0,0] + m[1] * Rotate[0,1] + m[2] * Rotate[0,2]
    result[1] = m[0] * Rotate[1,0] + m[1] * Rotate[1,1] + m[2] * Rotate[1,2]
    result[2] = m[0] * Rotate[2,0] + m[1] * Rotate[2,1] + m[2] * Rotate[2,2]
    result[3] = m[3]

proc scale*[T](m:Mat4x4[T], v:Vec3[T]): Mat4x4[T] =
    result = m
    result[0] = m[0] * v[0]
    result[1] = m[1] * v[1]
    result[2] = m[2] * v[2]

proc ortho*[T]( left, right, bottom, top, zNear, zFar:T): Mat4x4[T] =
    result = mat4[T](1.0)
    result[0,0] = T(2) / (right - left)
    result[1,1] = T(2) / (top - bottom)
    result[2,2] = -T(2) / (zFar - zNear)
    result[3,0] = -(right + left) / (right - left)
    result[3,1] = -(top + bottom) / (top - bottom)
    result[3,2] = -(zFar + zNear) / (zFar - zNear)
    result[3,3] = 1

proc perspectiveLH*[T]( fovy, aspect, zNear, zFar:T): Mat4x4[T] =
    let tanHalfFovy = tan(fovy / T(2))
    result = mat4[T](0.0)
    result[0,0] = T(1) / (aspect * tanHalfFovy)
    result[1,1] = T(1) / (tanHalfFovy)
    result[2,2] = (zFar + zNear) / (zFar - zNear)
    result[2,3] = T(1)
    result[3,2] = - (T(2) * zFar * zNear) / (zFar - zNear)

proc perspectiveRH*[T]( fovy, aspect, zNear, zFar:T): Mat4x4[T] =
    let tanHalfFovy = tan(fovy / T(2))
    result = mat4[T](0.0)
    result[0,0] = T(1) / (aspect * tanHalfFovy)
    result[1,1] = T(1) / (tanHalfFovy)
    result[2,3] = T(-1)

    result[2,2] = -(zFar + zNear) / (zFar - zNear)
    result[3,2] = -(T(2) * zFar * zNear) / (zFar - zNear)

proc lookAtRH*[T](eye,center,up:Vec3[T]): Mat4x4[T] =
    let
        f = normalize(center - eye)
        s = normalize(cross(f, up))
        u = cross(s,f)
    result = mat4[T](1.0)
    result.row0 = vec4( s,0)
    result.row1 = vec4( u,0)
    result.row2 = vec4(-f,0)
    result.arr[3] = vec4(-dot(s,eye), -dot(u,eye), dot(f,eye), 1)
    result[3,0] = -dot(s, eye)
    result[3,1] = -dot(u, eye)
    result[3,2] = dot(f, eye)

proc lookAtLH*[T](eye,center,up:Vec3[T]):Mat4x4[T]=
    let
        f = normalize(center - eye)
        s = normalize(cross(f, up))
        u = cross(s,f)
    result = mat4[T](1.0)

    result.row0 = vec4(s, 0)
    result.row1 = vec4(u, 0)
    result.row2 = vec4(f, 0)
    result.arr[3] = vec4(-dot(s, eye),-dot(u, eye),-dot(f, eye), 1)

proc frustum*[T](left, right, bottom, top, near, far: T): Mat4[T] =
  result[0][0] =       (2*near)/(right-left)
  result[1][1] =       (2*near)/(top-bottom)
  result[2][2] =     (far+near)/(near-far)
  result[2][0] =   (right+left)/(right-left)
  result[2][1] =   (top+bottom)/(top-bottom)
  result[2][3] = -1
  result[3][2] =   (2*far*near)/(near-far)


when GLM_LEFT_HAND:
    proc perspective*[T]( fovy, aspect, zNear, zFar:T):Mat4x4[T]=
        perspectiveLH(fovy, aspect, zNear, zFar)
    proc lookAt*[T](eye,center,up:Vec3[T]):Mat4x4[T]=
        lookAtLH(eye,center, up)
else:
    proc perspective*[T]( fovy, aspect, zNear, zFar:T):Mat4x4[T]=
        perspectiveRH(fovy, aspect, zNear, zFar)
    proc lookAt*[T](eye,center,up:Vec3[T]):Mat4x4[T]=
        lookAtRH(eye, center, up)

proc frustum*(left, right, bottom, top, near, far: SomeReal): Mat4[SomeReal] =
  result[0][0] =       (2*near)/(right-left)
  result[1][1] =       (2*near)/(top-bottom)
  result[2][2] =     (far+near)/(near-far)
  result[2][0] =   (right+left)/(right-left)
  result[2][1] =   (top+bottom)/(top-bottom)
  result[2][3] = -1
  result[3][2] =   (2*far*near)/(near-far)
        
if isMainModule:
    var m = mat4d()
    var nm = translate(m, vec3(5.0, 5.0, 5.0))
    var rm = rotate(m, vec3(0.0, 1.0, 0.0), radians(45.0) )
    var sm = scale(m, vec3(1.5, 5.0, 8.0))
    var v = vec4(1.0,0.0,0.0,1.0)
    var o = ortho(-5.0, 5, -5,5, 0.01, 100)
    var la = lookAt(vec3(0.0, 0,0), vec3(50.0, 50.0, 0.0), vec3(0.0, 1, 0.0))
    echo mat.`$`(la)

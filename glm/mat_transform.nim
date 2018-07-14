when not compiles(SomeFloat):
  type SomeFloat = SomeReal

import globals
import mat
import vec

proc translateInpl*[T](m:var Mat4[T]; v:Vec3[T]): void {.inline.} =
  m[3] = m * vec4(v,1)

proc translateInpl*[T](m: var Mat4[T]; x,y,z:T): void {.inline.} =
  m.translateInpl(vec3(x,y,z))

proc translate*[T](m:Mat4[T]; v:Vec3[T]): Mat4[T] {.inline.} =
  result = m
  result.translateInpl(v)

proc translate*[T](m:Mat4[T]; x,y,z:T): Mat4[T] {.inline.} =
  result = m
  result.translateInpl(vec3(x,y,z))

proc rotate*[T](m:Mat4x4[T]; angle:T, axis:Vec3[T]): Mat4x4[T] {.inline.} =
    let
      a = angle
      c = cos(a)
      s = sin(a)
    var
      axis = normalize(axis)
      temp  = T(1 - c) * axis
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

proc rotate*[T](m:Mat4x4[T]; axis:Vec3[T], angle:T): Mat4x4[T] {.deprecated.} =
  ## Please use the ``angle`` as first argument.
  m.rotate(angle,axis)

proc rotate*[T](m:Mat4x4[T]; angle,x,y,z:T): Mat4x4[T] = rotate(m, angle, vec3(x,y,z))
proc rotateX*[T](m:Mat4x4[T]; angle:T): Mat4x4[T] = rotate(m, angle, vec3[T](1,0,0))
proc rotateY*[T](m:Mat4x4[T]; angle:T): Mat4x4[T] = rotate(m, angle, vec3[T](0,1,0))
proc rotateZ*[T](m:Mat4x4[T]; angle:T): Mat4x4[T] = rotate(m, angle, vec3[T](0,0,1))

proc rotateInpl*[T](m:  var Mat4x4[T]; angle:T, axis:Vec3[T]): void {.inline.} =
  m = m.rotate(angle, axis)
proc rotateInpl*[T](m:  var Mat4x4[T]; angle,x,y,z: T): void {.inline.} =
  m = m.rotate(angle, vec3(x,y,z))
proc rotateInplX*[T](m: var Mat4x4[T]; angle:T): void {.inline.} =
  m = m.rotate(angle, vec3[T](1,0,0))
proc rotateInplY*[T](m: var Mat4x4[T]; angle:T): void {.inline.} =
  m = m.rotate(angle, vec3[T](0,1,0))
proc rotateInplZ*[T](m: var Mat4x4[T]; angle:T): void {.inline.} =
  m = m.rotate(angle, vec3[T](0,0,1))

proc scaleInpl*[T](m:var Mat4[T], v:Vec3[T]): void {.inline.} =
  m[0] *= v[0]
  m[1] *= v[1]
  m[2] *= v[2]

proc scaleInpl*[T](m:var Mat4[T], x,y,z:T): void {.inline.} =
  m[0] *= x
  m[1] *= y
  m[2] *= z

proc scaleInpl*[T](m:var Mat4[T], s: T): void {.inline.} =
  m[0] *= s
  m[1] *= s
  m[2] *= s

proc scale*[T](m:Mat4[T], v:Vec3[T]): Mat4x4[T] {.inline.} =
  result = m
  result.scaleInpl(v)

proc scale*[T](m:Mat4[T], x,y,z: T): Mat4[T] {.inline.} =
  result = m
  result.scaleInpl(x,y,z)

proc scale*[T](m:Mat4[T], s: T): Mat4[T] {.inline.} =
  result = m
  result.scaleInpl(s)

proc pickMatrix*[T](center, delta: Vec2[T]; viewport: Vec4[T]): Mat4[T] =
  ## Define a picking region
  assert(delta.x > T(0) and delta.y > T(0))
  result = mat4(T(1))
  if not (delta.x > T(0) and delta.y > T(0)):
    return # Error

  let Temp: Vec3[T] = vec3(
    (viewport[2] - T(2) * (center.x - viewport[0])) / delta.x,
    (viewport[3] - T(2) * (center.y - viewport[1])) / delta.y,
    T(0)
  )

  # Translate and scale the picked region to the entire window
  result = translate(result, Temp)
  result = scale(result, vec3(viewport[2] / delta.x, viewport[3] / delta.y, T(1)))

proc ortho*[T]( left, right, bottom, top, zNear, zFar:T): Mat4[T] =
    result = mat4[T](1.0)
    result[0,0] = T(2) / (right - left)
    result[1,1] = T(2) / (top - bottom)
    result[2,2] = -T(2) / (zFar - zNear)
    result[3,0] = -(right + left) / (right - left)
    result[3,1] = -(top + bottom) / (top - bottom)
    result[3,2] = -(zFar + zNear) / (zFar - zNear)
    result[3,3] = 1

proc perspectiveLH*[T]( fovy, aspect, zNear, zFar:T): Mat4[T] =
    let tanHalfFovy = tan(fovy / T(2))
    result = mat4[T](0.0)
    result[0,0] = T(1) / (aspect * tanHalfFovy)
    result[1,1] = T(1) / (tanHalfFovy)
    result[2,2] = (zFar + zNear) / (zFar - zNear)
    result[2,3] = T(1)
    result[3,2] = - (T(2) * zFar * zNear) / (zFar - zNear)

proc perspectiveRH*[T]( fovy, aspect, zNear, zFar:T): Mat4[T] =
    let tanHalfFovy = tan(fovy / T(2))
    result = mat4[T](0.0)
    result[0,0] = T(1) / (aspect * tanHalfFovy)
    result[1,1] = T(1) / (tanHalfFovy)
    result[2,3] = T(-1)

    result[2,2] = -(zFar + zNear) / (zFar - zNear)
    result[3,2] = -(T(2) * zFar * zNear) / (zFar - zNear)

proc lookAtRH*[T](eye,center,up:Vec3[T]): Mat4[T] =
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

proc lookAtLH*[T](eye,center,up:Vec3[T]):Mat4[T]=
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
    proc perspective*[T]( fovy, aspect, zNear, zFar:T):Mat4[T]=
        perspectiveLH(fovy, aspect, zNear, zFar)
    proc lookAt*[T](eye,center,up:Vec3[T]):Mat4[T]=
        lookAtLH(eye,center, up)
else:
    proc perspective*[T]( fovy, aspect, zNear, zFar:T):Mat4[T]=
        perspectiveRH(fovy, aspect, zNear, zFar)
    proc lookAt*[T](eye,center,up:Vec3[T]):Mat4[T]=
        lookAtRH(eye, center, up)

proc frustum*(left, right, bottom, top, near, far: SomeFloat): Mat4[SomeFloat] =
  result[0][0] =       (2*near)/(right-left)
  result[1][1] =       (2*near)/(top-bottom)
  result[2][2] =     (far+near)/(near-far)
  result[2][0] =   (right+left)/(right-left)
  result[2][1] =   (top+bottom)/(top-bottom)
  result[2][3] = -1
  result[3][2] =   (2*far*near)/(near-far)

when isMainModule:
    var m = mat4d()
    var nm = translate(m, vec3(5.0, 5.0, 5.0))
    var rm = rotate(m, vec3(0.0, 1.0, 0.0), radians(45.0) )
    var sm = scale(m, vec3(1.5, 5.0, 8.0))
    var v = vec4(1.0,0.0,0.0,1.0)
    var o = ortho(-5.0, 5, -5,5, 0.01, 100)
    var la = lookAt(vec3(0.0, 0,0), vec3(50.0, 50.0, 0.0), vec3(0.0, 1, 0.0))
    echo mat.`$`(la)

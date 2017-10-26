import glm
import unittest

proc epsilonEqual(a,b,epsilon: float32): bool =
  return abs(a - b) < epsilon

proc epsilonEqual(a,b: Quatf; epsilon: float32): Vec4b =
  for i, x in a - b:
    result[i] = abs(x) < epsilon

proc epsilonEqual[N,T](a,b: Vec[N,T]; epsilon: T): Vec[N,bool] =
  for i, x in (a - b).arr:
    result[i] = abs(x) < epsilon

proc epsilonEqual(a,b: Mat4f; epsilon: float32): Mat4[bool] =
  for i in 0..<4:
    result[i] = epsilonEqual(a.arr[i], b.arr[i], epsilon)

proc angleAxis(angle: float32, axis: Vec3f): Quatf =
  quatf(axis, angle)

proc all[N,M](arg: Mat[N,M,bool]): bool =
  for x in arg.arr:
    if not all(x):
      return false
  return true

suite "quaternion":
  test "angle":
    block:
      let Q: Quatf = angleAxis(float32(Pi) * 0.25f, vec3f(0, 0, 1));
      let N: Quatf = glm.normalize(Q);
      let L: float32 = glm.length(N);
      check epsilonEqual(L, 1.0f, 0.01f)
      let A: float32 = glm.angle(N);
      check epsilonEqual(A, float32(Pi) * 0.25f, 0.01f)

    block:
      let Q: Quatf = angleAxis(float32(Pi) * 0.25f, glm.normalize(glm.vec3f(0, 1, 1)));
      let N: Quatf = glm.normalize(Q);
      let L: float32 = glm.length(N);
      check epsilonEqual(L, 1.0f, 0.01f)
      let A: float32 = glm.angle(N);
      check epsilonEqual(A, float32(Pi) * 0.25f, 0.01f)

    block:
      let Q: Quatf = angleAxis(float32(Pi) * 0.25f, glm.normalize(glm.vec3f(1, 2, 3)));
      let N: Quatf = glm.normalize(Q);
      let L: float32 = glm.length(N);
      check epsilonEqual(L, 1.0f, 0.01f)
      let A: float32 = glm.angle(N);
      check epsilonEqual(A, float32(Pi) * 0.25f, 0.01f)

  test "angleAxis":
    let A: Quatf = angleAxis(0.0f, vec3f(0, 0, 1));
    let B: Quatf = angleAxis(float32(Pi) * 0.5f, vec3f(0, 0, 1));
    let C: Quatf = glm.mix(A, B, 0.5f);
    let D: Quatf = angleAxis(float32(Pi) * 0.25f, vec3f(0, 0, 1));

    check epsilonEqual(C.x, D.x, 0.01f)
    check epsilonEqual(C.y, D.y, 0.01f)
    check epsilonEqual(C.z, D.z, 0.01f)
    check epsilonEqual(C.w, D.w, 0.01f)

  test "mix":
    let A: Quatf = angleAxis(0.0f, vec3f(0, 0, 1));
    let B: Quatf = angleAxis(float32(Pi) * 0.5f, vec3f(0, 0, 1));
    let C: Quatf = glm.mix(A, B, 0.5f);
    let D: Quatf = angleAxis(float32(Pi) * 0.25f, vec3f(0, 0, 1));

    check epsilonEqual(C.x, D.x, 0.01f)
    check epsilonEqual(C.y, D.y, 0.01f)
    check epsilonEqual(C.z, D.z, 0.01f)
    check epsilonEqual(C.w, D.w, 0.01f)

  test "normalize":
    block:
      let Q: Quatf = angleAxis(float32(Pi) * 0.25f, vec3f(0, 0, 1));
      let N: Quatf = glm.normalize(Q);
      let L: float32 = glm.length(N);
      check epsilonEqual(L, 1.0f, 0.000001f)

    block:
      let Q: Quatf = angleAxis(float32(Pi) * 0.25f, vec3f(0, 0, 2));
      let N: Quatf = glm.normalize(Q);
      let L: float32 = glm.length(N);
      check epsilonEqual(L, 1.0f, 0.000001f)

    block:
      let Q: Quatf = angleAxis(float32(Pi) * 0.25f, vec3f(1, 2, 3));
      let N: Quatf = glm.normalize(Q);
      let L: float32 = glm.length(N);
      check epsilonEqual(L, 1.0f, 0.000001f)

  test "euler":
    block:
      let q = quatf(0,0,1,1)
      let Roll: float32  = glm.roll(q);
      let Pitch: float32 = glm.pitch(q);
      let Yaw: float32   = glm.yaw(q);
      let Angles: Vec3f  = glm.eulerAngles(q);
      check Angles.x == q.yaw
      check Angles.y == q.pitch
      check Angles.z == q.roll

    block:
      let q = quatd(0,0,1,1)
      let Roll: float64 = glm.roll(q);
      let Pitch: float64 = glm.pitch(q);
      let Yaw: float64 = glm.yaw(q);
      let Angles: Vec3d = glm.eulerAngles(q);

  test "slerp":
    const Epsilon = 0.01f;
    const sqrt2 = sqrt(2.0f) / 2.0f;

    let id = quatf()
    let Y90rot: Quatf = quatf(0.0f, sqrt2, 0.0f, sqrt2);
    let Y180rot: Quatf = quatf(0.0f, 1.0f, 0.0f, 0.0f);

    # Testing a == 0
    # Must be id
    let id2: Quatf = slerp(id, Y90rot, 0.0f);
    check glm.all(epsilonEqual(id, id2, Epsilon))

    # Testing a == 1
    # Must be 90° rotation on Y : 0 0.7 0 0.7
    let Y90rot2: Quatf = slerp(id, Y90rot, 1.0f);
    check glm.all(epsilonEqual(Y90rot, Y90rot2, Epsilon))

    # Testing standard, easy case
    # Must be 45° rotation on Y : 0 0.38 0 0.92
    let Y45rot1: Quatf = slerp(id, Y90rot, 0.5f);

    # Testing reverse case
    # Must be 45° rotation on Y : 0 0.38 0 0.92
    let Ym45rot2: Quatf = slerp(Y90rot, id, 0.5f);
    check all(epsilonEqual(Y45rot1, Ym45rot2, Epsilon))

    # Testing against full circle around the sphere instead of shortest path
    # Must be 45° rotation on Y
    # certainly not a 135° rotation

    let Y45rot3: Quatf = slerp(id , -Y90rot, 0.5f);
    let Y45angle3: float32 = glm.angle(Y45rot3);

    check epsilonEqual(Y45angle3, radians(45.0f), Epsilon)
    check all(epsilonEqual(Ym45rot2, Y45rot3, Epsilon))

    # Same, but inverted
    # Must also be 45° rotation on Y :  0 0.38 0 0.92
    # -0 -0.38 -0 -0.92 is ok too
    let Y45rot4: Quatf = slerp(-Y90rot, id, 0.5f);
    check glm.all(epsilonEqual(Ym45rot2, -Y45rot4, Epsilon))

    # Testing q1 = q2
    # Must be 90° rotation on Y : 0 0.7 0 0.7
    let Y90rot3: Quatf = slerp(Y90rot, Y90rot, 0.5f);
    check glm.all(epsilonEqual(Y90rot, Y90rot3, Epsilon))

    # Testing 180° rotation
    # Must be 90° rotation on almost any axis that is on the XZ plane
    let XZ90rot: Quatf = slerp(id, -Y90rot, 0.5f);
    let XZ90angle: float32 = glm.angle(XZ90rot); # Must be PI/4 = 0.78;
    check epsilonEqual(XZ90angle, float32(Pi) * 0.25f, Epsilon)

    # Testing almost equal quaternions (this test should pass through the linear interpolation)
    # Must be 0 0.00X 0 0.99999
    let almostid: Quatf = slerp(id, angleAxis(0.1f, vec3f(0.0f, 1.0f, 0.0f)), 0.5f);

    # Testing quaternions with opposite sign
    block:
      let a: Quatf = quatf(0, 0, 0, -1);

      let result: Quatf = slerp(a, id, 0.5f);
      check epsilonEqual(glm.pow(glm.dot(id, result), 2.0f), 1.0f, 0.01f)

  test "mul":
    let temp1: Quatf = normalize(quatf(vec3f(0.0, 1.0, 0.0), 1.0f));
    let temp2: Quatf = normalize(quatf(vec3f(1.0, 0.0, 0.0), 0.5f));

    let transformed0: Vec3f = temp1 * vec3f(0.0, 1.0, 0.0) * inverse(temp1);
    let temp4: Vec3f = temp2 * transformed0 * inverse(temp2);

    let temp5: Quatf = normalize(temp1 * temp2);
    let temp6: Vec3f = temp5 * vec3f(0.0, 1.0, 0.0) * inverse(temp5);

    var temp7 = quatf();
    temp7 *= temp5;
    temp7 *= inverse(temp5);

    check all(epsilonEqual(temp7, quatf(), 0.01f))

  test "two axis ctr":
    let q1: Quatf = quatf(vec3f(1, 0, 0), vec3f(0, 1, 0));
    let v1: Vec3f = q1 * vec3f(1, 0, 0);
    check all(epsilonEqual(v1, vec3f(0, 1, 0), 0.0001f))

    let q2: Quatf = q1 * q1;
    let v2: Vec3f = q2 * vec3f(1, 0, 0);
    check all(epsilonEqual(v2, vec3f(-1, 0, 0), 0.0001f))

  test "type":
    var A: Quatf
    var B: Quatd


  test "mul vec":
    let q: Quatf = angleAxis(float32(Pi) * 0.5f, vec3f(0, 0, 1))
    let v = vec3f(1, 0, 0)
    let u = vec3f(q * v)
    let w = vec3f(u * q)

    check all(epsilonEqual(v, w, 0.01f))

  test "size":
     check 16 == sizeof(Quatf)
     check 32 == sizeof(Quatd)

  test "mat convert":
    let a = 1.0f / sqrt(2.0f)
    let b = 1.0f / sqrt(3.0f)

    for q in [quatf(1,0,0,0), quatf(0,1,0,0), quatf(0,0,1,0), quatf(0,0,0,1),
              quatf(a,a,0,0), quatf(a,0,a,0), quatf(a,0,0,a), quatf(0,a,a,0), quatf(0,a,0,a), quatf(0,0,a,a),
              quatf(0,b,b,b), quatf(b,0,b,b), quatf(b,b,0,b), quatf(b,b,b,0)]:
      let mat = mat3(q)
      let q2 = quat(mat3(q))
      check all(epsilonEqual(q,q2, 0.01f))

  test "rotate":
    let HalfPi = float32(PI * 0.5)

    let data = [
      (HalfPi, vec3f(1,0,0)),
      (HalfPi, vec3f(0,1,0)),
      (HalfPi, vec3f(0,0,1)),
      (HalfPi, vec3f(-1,0,0)),
      (HalfPi, vec3f(0,-1,0)),
      (HalfPi, vec3f(0,0,-1)),
      (0.123f,  vec3f(4,-2,-8))
    ]

    for angle, axis in data.items:
      let m1 =
        quatf()
        .rotate(angle, axis)
        .mat4f

      let m2 =
        mat4f(1)
        .rotate(angle, axis)

      check all(epsilonEqual(m1,m2, 0.001f))


    let m1 =
      quatf()
      .rotate(HalfPi, vec3f(1,0,0))
      .rotate(HalfPi, vec3f(0,1,0))
      .mat4f

    let m2 =
      mat4f(1)
      .rotate(HalfPi, vec3f(1,0,0))
      .rotate(HalfPi, vec3f(0,1,0))

    check all(epsilonEqual(m1,m2, 0.001f))

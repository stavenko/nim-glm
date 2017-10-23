import ../glm


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

proc test_quat_angle(): int =
  var Error: int = 0;

  block:
    let Q: Quatf = angleAxis(float32(Pi) * 0.25f, vec3f(0, 0, 1));
    let N: Quatf = glm.normalize(Q);
    let L: float32 = glm.length(N);
    Error += (if epsilonEqual(L, 1.0f, 0.01f): 0 else: 1)
    let A: float32 = glm.angle(N);
    Error += (if epsilonEqual(A, float32(Pi) * 0.25f, 0.01f): 0 else: 1)

  block:
    let Q: Quatf = angleAxis(float32(Pi) * 0.25f, glm.normalize(glm.vec3f(0, 1, 1)));
    let N: Quatf = glm.normalize(Q);
    let L: float32 = glm.length(N);
    Error += (if epsilonEqual(L, 1.0f, 0.01f): 0 else: 1)
    let A: float32 = glm.angle(N);
    Error += (if epsilonEqual(A, float32(Pi) * 0.25f, 0.01f): 0 else: 1)

  block:
    let Q: Quatf = angleAxis(float32(Pi) * 0.25f, glm.normalize(glm.vec3f(1, 2, 3)));
    let N: Quatf = glm.normalize(Q);
    let L: float32 = glm.length(N);
    Error += (if epsilonEqual(L, 1.0f, 0.01f): 0 else: 1)
    let A: float32 = glm.angle(N);
    Error += (if epsilonEqual(A, float32(Pi) * 0.25f, 0.01f): 0 else: 1)


  return Error;

proc test_quat_angleAxis(): int =
  var Error: int = 0;

  let A: Quatf = angleAxis(0.0f, vec3f(0, 0, 1));
  let B: Quatf = angleAxis(float32(Pi) * 0.5f, vec3f(0, 0, 1));
  let C: Quatf = glm.mix(A, B, 0.5f);
  let D: Quatf = angleAxis(float32(Pi) * 0.25f, vec3f(0, 0, 1));

  Error += (if epsilonEqual(C.x, D.x, 0.01f): 0 else: 1)
  Error += (if epsilonEqual(C.y, D.y, 0.01f): 0 else: 1)
  Error += (if epsilonEqual(C.z, D.z, 0.01f): 0 else: 1)
  Error += (if epsilonEqual(C.w, D.w, 0.01f): 0 else: 1)

  return Error;

proc test_quat_mix(): int =
  var Error: int = 0;

  let A: Quatf = angleAxis(0.0f, vec3f(0, 0, 1));
  let B: Quatf = angleAxis(float32(Pi) * 0.5f, vec3f(0, 0, 1));
  let C: Quatf = glm.mix(A, B, 0.5f);
  let D: Quatf = angleAxis(float32(Pi) * 0.25f, vec3f(0, 0, 1));

  Error += (if epsilonEqual(C.x, D.x, 0.01f): 0 else: 1)
  Error += (if epsilonEqual(C.y, D.y, 0.01f): 0 else: 1)
  Error += (if epsilonEqual(C.z, D.z, 0.01f): 0 else: 1)
  Error += (if epsilonEqual(C.w, D.w, 0.01f): 0 else: 1)

  return Error;

proc test_quat_normalize(): int =
  var Error: int = 0;

  block:
    let Q: Quatf = angleAxis(float32(Pi) * 0.25f, vec3f(0, 0, 1));
    let N: Quatf = glm.normalize(Q);
    let L: float32 = glm.length(N);
    Error += (if epsilonEqual(L, 1.0f, 0.000001f): 0 else: 1)

  block:
    let Q: Quatf = angleAxis(float32(Pi) * 0.25f, vec3f(0, 0, 2));
    let N: Quatf = glm.normalize(Q);
    let L: float32 = glm.length(N);
    Error += (if epsilonEqual(L, 1.0f, 0.000001f): 0 else: 1)

  block:
    let Q: Quatf = angleAxis(float32(Pi) * 0.25f, vec3f(1, 2, 3));
    let N: Quatf = glm.normalize(Q);
    let L: float32 = glm.length(N);
    Error += (if epsilonEqual(L, 1.0f, 0.000001f): 0 else: 1)


  return Error;

proc test_quat_euler(): int =
  var Error: int = 0;

  block:
    let q = quatf(0,0,1,1)
    let Roll: float32  = glm.roll(q);
    let Pitch: float32 = glm.pitch(q);
    let Yaw: float32   = glm.yaw(q);
    let Angles: Vec3f  = glm.eulerAngles(q);

  block:
    let q = quatd(0,0,1,1)
    let Roll: float64 = glm.roll(q);
    let Pitch: float64 = glm.pitch(q);
    let Yaw: float64 = glm.yaw(q);
    let Angles: Vec3d = glm.eulerAngles(q);

  return Error;

proc test_quat_slerp(): int =
  var Error: int = 0;

  const Epsilon = 0.01f;
  const sqrt2 = sqrt(2.0f) / 2.0f;

  let id = quatf()
  let Y90rot: Quatf = quatf(0.0f, sqrt2, 0.0f, sqrt2);
  let Y180rot: Quatf = quatf(0.0f, 1.0f, 0.0f, 0.0f);

  # Testing a == 0
  # Must be id
  let id2: Quatf = slerp(id, Y90rot, 0.0f);
  Error += (if glm.all(epsilonEqual(id, id2, Epsilon)): 0 else: 1)

  # Testing a == 1
  # Must be 90° rotation on Y : 0 0.7 0 0.7
  let Y90rot2: Quatf = slerp(id, Y90rot, 1.0f);
  Error += (if glm.all(epsilonEqual(Y90rot, Y90rot2, Epsilon)): 0 else: 1)

  # Testing standard, easy case
  # Must be 45° rotation on Y : 0 0.38 0 0.92
  let Y45rot1: Quatf = slerp(id, Y90rot, 0.5f);

  # Testing reverse case
  # Must be 45° rotation on Y : 0 0.38 0 0.92
  let Ym45rot2: Quatf = slerp(Y90rot, id, 0.5f);
  Error += (if all(epsilonEqual(Y45rot1, Ym45rot2, Epsilon)): 0 else: 1)

  # Testing against full circle around the sphere instead of shortest path
  # Must be 45° rotation on Y
  # certainly not a 135° rotation

  let Y45rot3: Quatf = slerp(id , -Y90rot, 0.5f);
  let Y45angle3: float32 = glm.angle(Y45rot3);

  Error += (if epsilonEqual(Y45angle3, radians(45.0f), Epsilon): 0 else: 1)
  Error += (if all(epsilonEqual(Ym45rot2, Y45rot3, Epsilon)): 0 else: 1)

  # Same, but inverted
  # Must also be 45° rotation on Y :  0 0.38 0 0.92
  # -0 -0.38 -0 -0.92 is ok too
  let Y45rot4: Quatf = slerp(-Y90rot, id, 0.5f);
  Error += (if glm.all(epsilonEqual(Ym45rot2, -Y45rot4, Epsilon)): 0 else: 1)

  # Testing q1 = q2
  # Must be 90° rotation on Y : 0 0.7 0 0.7
  let Y90rot3: Quatf = slerp(Y90rot, Y90rot, 0.5f);
  Error += (if glm.all(epsilonEqual(Y90rot, Y90rot3, Epsilon)): 0 else: 1)

  # Testing 180° rotation
  # Must be 90° rotation on almost any axis that is on the XZ plane
  let XZ90rot: Quatf = slerp(id, -Y90rot, 0.5f);
  let XZ90angle: float32 = glm.angle(XZ90rot); # Must be PI/4 = 0.78;
  Error += (if epsilonEqual(XZ90angle, float32(Pi) * 0.25f, Epsilon): 0 else: 1)

  # Testing almost equal quaternions (this test should pass through the linear interpolation)
  # Must be 0 0.00X 0 0.99999
  let almostid: Quatf = slerp(id, angleAxis(0.1f, vec3f(0.0f, 1.0f, 0.0f)), 0.5f);

  # Testing quaternions with opposite sign
  block:
    let a: Quatf = quatf(0, 0, 0, -1);

    let result: Quatf = slerp(a, id, 0.5f);
    Error += (if epsilonEqual(glm.pow(glm.dot(id, result), 2.0f), 1.0f, 0.01f): 0 else: 1)


  return Error;

proc test_quat_mul(): int =
  var Error: int = 0;

  let temp1: Quatf = normalize(quatf(vec3f(0.0, 1.0, 0.0), 1.0f));
  let temp2: Quatf = normalize(quatf(vec3f(1.0, 0.0, 0.0), 0.5f));

  let transformed0: Vec3f = temp1 * vec3f(0.0, 1.0, 0.0) * inverse(temp1);
  let temp4: Vec3f = temp2 * transformed0 * inverse(temp2);

  let temp5: Quatf = normalize(temp1 * temp2);
  let temp6: Vec3f = temp5 * vec3f(0.0, 1.0, 0.0) * inverse(temp5);

  var temp7 = quatf();
  temp7 *= temp5;
  temp7 *= inverse(temp5);

  Error += (if all(epsilonEqual(temp7, quatf(), 0.01f)): 0 else: 1)

  return Error;

proc test_quat_two_axis_ctr(): int =
  var Error: int = 0;

  let q1: Quatf = quatf(vec3f(1, 0, 0), vec3f(0, 1, 0));
  let v1: Vec3f = q1 * vec3f(1, 0, 0);
  Error += (if all(epsilonEqual(v1, vec3f(0, 1, 0), 0.0001f)): 0 else: 1)

  let q2: Quatf = q1 * q1;
  let v2: Vec3f = q2 * vec3f(1, 0, 0);
  Error += (if all(epsilonEqual(v2, vec3f(-1, 0, 0), 0.0001f)): 0 else: 1)

  return Error;

proc test_quat_type(): int =
  var A: Quatf
  var B: Quatd
  return 0;

proc test_quat_mul_vec(): int =
  var Error: int = 0;

  let q: Quatf = angleAxis(float32(Pi) * 0.5f, vec3f(0, 0, 1))
  let v = vec3f(1, 0, 0)
  let u = vec3f(q * v)
  let w = vec3f(u * q)

  Error += (if all(epsilonEqual(v, w, 0.01f)): 0 else: 1)

  return Error;

proc test_size(): int =
   var Error: int = 0;

   Error += (if 16 == sizeof(Quatf): 0 else: 1)
   Error += (if 32 == sizeof(Quatd): 0 else: 1)

   return Error;

proc test_quat_mat_convert(): int =
  var Error = 0;

  let a = 1.0f / sqrt(2.0f)
  let b = 1.0f / sqrt(3.0f)

  for q in [quatf(1,0,0,0), quatf(0,1,0,0), quatf(0,0,1,0), quatf(0,0,0,1),
            quatf(a,a,0,0), quatf(a,0,a,0), quatf(a,0,0,a), quatf(0,a,a,0), quatf(0,a,0,a), quatf(0,0,a,a),
            quatf(0,b,b,b), quatf(b,0,b,b), quatf(b,b,0,b), quatf(b,b,b,0)]:
    let mat = mat3(q)
    let q2 = quat(mat3(q))
    Error += (if all(epsilonEqual(q,q2, 0.01f)): 0 else: 1)

  Error

var Error = 0;

#Error += test_quat_ctr();
Error += test_quat_mul_vec()
Error += test_quat_two_axis_ctr()
Error += test_quat_mul()
#Error += test_quat_precision()
Error += test_quat_type()
Error += test_quat_angle()
Error += test_quat_angleAxis()
Error += test_quat_mix()
Error += test_quat_normalize()
Error += test_quat_euler()
Error += test_quat_slerp()
Error += test_size()
Error += test_quat_mat_convert()

if Error != 0:
  quit(Error)

block test_quat_rotate:
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

    doAssert all(epsilonEqual(m1,m2, 0.001f))


  let m1 =
    quatf()
    .rotate(HalfPi, vec3f(1,0,0))
    .rotate(HalfPi, vec3f(0,1,0))
    .mat4f

  let m2 =
    mat4f(1)
    .rotate(HalfPi, vec3f(1,0,0))
    .rotate(HalfPi, vec3f(0,1,0))

  doAssert all(epsilonEqual(m1,m2, 0.001f))

proc rotate*[T](q: Quat[T]; angle: T; v: Vec3[T]) : Quat[T] =
  ## rotates q around the axis v by the given angle in radians
  result.arr = vec4(
    normalize(v) * sin(angle * 0.5),
    cos(angle * 0.5)
  ).arr
  result = q * result

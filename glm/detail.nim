when not compiles(SomeFloat):
  type SomeFloat = SomeReal

import vec

proc mod289*[T: SomeFloat](x: T): T =
  return x - floor(x * T(1.0) / T(289.0)) * T(289.0);

proc mod289*[N, T](x: Vec[N,T]): Vec[N,T] =
  return x - floor(x * T(1.0) / T(289.0)) * T(289.0);

proc permute*[T](x: T): T =
  return mod289(((x * T(34)) + T(1)) * x);

proc permute*[T](x: Vec2[T]): Vec2[T] =
  return mod289(((x * T(34)) + T(1)) * x);

proc permute*[T](x: Vec3[T]):  Vec3[T] =
  return mod289(((x * T(34)) + T(1)) * x);

proc permute*[T](x: Vec4[T]):  Vec4[T] =
  return mod289(((x * T(34)) + T(1)) * x);

proc permute*(x: Vec): Vec =
  return mod289(((x * Vec.T(34)) + Vec.T(1)) * x);

proc taylorInvSqrt*[T](r: T):  T =
  return T(1.79284291400159) - T(0.85373472095314) * r;

proc taylorInvSqrt*[T](r: Vec2[T]):  Vec2[T] =
  return T(1.79284291400159) - T(0.85373472095314) * r;

proc taylorInvSqrt*[T](r: Vec3[T]):  Vec3[T] =
  return T(1.79284291400159) - T(0.85373472095314) * r;

proc taylorInvSqrt*[T](r: Vec4[T]):  Vec4[T] =
  return T(1.79284291400159) - T(0.85373472095314) * r;

proc taylorInvSqrt*[N, T](r: Vec[N, T]):  Vec[N, T] =
  return T(1.79284291400159) - T(0.85373472095314) * r;

proc fade*[T](t: Vec2[T]):  Vec2[T] =
  return (t * t * t) * (t * (t * T(6) - T(15)) + T(10));

proc fade*[T](t: Vec3[T]):  Vec3[T] =
  return (t * t * t) * (t * (t * T(6) - T(15)) + T(10));

proc fade*[T](t: Vec4[T]):  Vec4[T] =
  return (t * t * t) * (t * (t * T(6) - T(15)) + T(10));

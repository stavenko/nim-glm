when not compiles(SomeFloat):
  type SomeFloat = SomeReal

import vec

proc mod289*[T: SomeFloat](x: T): T {.noinit.} =
  return x - floor(x * T(1.0) / T(289.0)) * T(289.0);

proc mod289*[N, T](x: Vec[N,T]): Vec[N,T] {.noinit.} =
  return x - floor(x * T(1.0) / T(289.0)) * T(289.0);

proc permute*[T: SomeFloat](x: T): T {.noinit.} =
  return mod289(((x * T(34)) + T(1)) * x)

proc permute*[N,T](x: Vec[N,T]): Vec[N,T] {.noinit.} =
  return mod289(((x * T(34)) + T(1)) * x)

proc taylorInvSqrt*[T: SomeFloat](r: T): T {.noinit.} =
  return T(1.79284291400159) - T(0.85373472095314) * r

proc taylorInvSqrt*[N, T](r: Vec[N, T]):  Vec[N, T] {.noinit.} =
  return T(1.79284291400159) - T(0.85373472095314) * r

proc fade*[T: SomeFloat](t: T): T {.noinit.} =
  return (t * t * t) * (t * (t * T(6) - T(15)) + T(10))

proc fade*[N,T](t: Vec[N,T]): Vec[N,T] {.noinit.} =
  return (t * t * t) * (t * (t * T(6) - T(15)) + T(10))

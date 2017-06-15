import mat
import vec

#[
proc catmullRom*[N,T](v1,v2,v3,v4: Vec[N,T]; s: T): Vec[N,T] =
  ## return a point from a catmull rom curve
  let s1 = s
  let s2 = s*s
  let s3 = s*s2

  let f1 = -s3 + T(2) * s2 - s
  let f2 = T(3) * s3 - T(5) * s2 + T(2)
  let f3 = T(-3) * s3 + T(4) * s2 + s
  let f4 = s3 - s2

  return (f1 * v1 + f2 * v2 + f3 * v3 + f4 * v4) / T(2)


proc    hermite*[N,T](v1,v2,v3,v4: Vec[N,T]; s: T): Vec[N,T] =
  ## return a point from a hermite curve
  let s1 = s;
  let s2 = s*s;
  let s3 = s*s2;
  let f1 = T(2) * s3 - T(3) * s2 + T(1);
  let f2 = T(-2) * s3 + T(3) * s2;
  let f3 = s3 - T(2) * s2 + s;
  let f4 = s3 - s2;

  return f1 * v1 + f2 * v2 + f3 * t1 + f4 * t2

proc      cubic*[N,T](v1,v2,v3,v4: Vec[N,T]; s: T): Vec[N,T] =
  ## return a point from a cubic curve
	return ((v1 * s + v2) * s + v3) * s + v4
]#

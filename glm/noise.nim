
import vec
import detail

# Based on the work of Stefan Gustavson and Ashima Arts on "webgl-noise":
# https://github.com/ashima/webgl-noise
# Following Stefan Gustavson's paper "Simplex noise demystified":
# http://www.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf

proc gtc_grad4[T](j: T; ip: Vec4[T]): Vec4[T] =
  var pXYZ: Vec3[T] = floor(fract(vec3(j) * vec3[T](ip)) * T(7)) * ip[2] - T(1);
  let pW: T = T(1.5) - dot(abs(pXYZ), vec3[T](1));
  let s: Vec4[T] = vec4[T](lessThan(vec4[T](pXYZ, pW), vec4[T](0.0)));
  pXYZ = pXYZ + (vec3[T](s) * T(2) - T(1)) * s.w;
  return vec4(pXYZ, pW);

proc perlin*[T](Position: Vec2[T]): T =
  ## Clissic Perlin noise
  let Pi: Vec4[T]  = floor(Vec4[T](Position.x, Position.y, Position.x, Position.y)) + Vec4[T](0.0, 0.0, 1.0, 1.0);
  let Pf: Vec4[T] = fract(Vec4[T](Position.x, Position.y, Position.x, Position.y)) - Vec4[T](0.0, 0.0, 1.0, 1.0);
  Pi = mod(Pi, Vec4[T](289)); # To avoid truncation effects in permutation
  let ix = vec4(Pi.x, Pi.z, Pi.x, Pi.z);
  let iy = vec4(Pi.y, Pi.y, Pi.w, Pi.w);
  let fx = vec4(Pf.x, Pf.z, Pf.x, Pf.z);
  let fy = vec4(Pf.y, Pf.y, Pf.w, Pf.w);

  let i: Vec4[T] = detail.permute(detail.permute(ix) + iy);

  let gx: Vec4[T] = T(2) * fract(i / T(41)) - T(1);
  let gy: Vec4[T] = abs(gx) - T(0.5);
  let tx: Vec4[T] = floor(gx + T(0.5));
  gx = gx - tx;

  var g00 = vec2(gx.x, gy.x);
  var g10 = vec2(gx.y, gy.y);
  var g01 = vec2(gx.z, gy.z);
  var g11 = vec2(gx.w, gy.w);

  let norm: Vec4[T] = detail.taylorInvSqrt(Vec4[T](dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
  g00 *= norm.x;
  g01 *= norm.y;
  g10 *= norm.z;
  g11 *= norm.w;

  let n00: T = dot(g00, Vec2[T](fx.x, fy.x));
  let n10: T = dot(g10, Vec2[T](fx.y, fy.y));
  let n01: T = dot(g01, Vec2[T](fx.z, fy.z));
  let n11: T = dot(g11, Vec2[T](fx.w, fy.w));

  let fade_xy: Vec2[T] = detail.fade(Vec2[T](Pf.x, Pf.y));
  let n_x: Vec2[T] = mix(Vec2[T](n00, n01), Vec2[T](n10, n11), fade_xy.x);
  let n_xy: T = mix(n_x.x, n_x.y, fade_xy.y);
  return T(2.3) * n_xy;

proc perlin*[T](Position: Vec3[T]): T =
  ## Classic Perlin noise

  let Pi0: Vec3[T] = floor(Position); # Integer part for indexing
  let Pi1: Vec3[T] = Pi0 + T(1); # Integer part + 1
  Pi0 = detail.mod289(Pi0);
  Pi1 = detail.mod289(Pi1);
  let Pf0: Vec3[T] = fract(Position); # Fractional part for interpolation
  let Pf1: Vec3[T] = Pf0 - T(1); # Fractional part - 1.0
  let ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x)
  let iy: Vec4[T] = Vec4[T](Vec2[T](Pi0.y), Vec2[T](Pi1.y));
  let iz0 = vec4(Pi0.z);
  let iz1 = vec4(Pi1.z);

  let ixy: Vec4[T] = detail.permute(detail.permute(ix) + iy);
  let ixy0: Vec4[T] = detail.permute(ixy + iz0);
  let ixy1: Vec4[T] = detail.permute(ixy + iz1);

  let gx0: Vec4[T] = ixy0 * T(1.0 / 7.0);
  let gy0: Vec4[T] = fract(floor(gx0) * T(1.0 / 7.0)) - T(0.5);
  gx0 = fract(gx0);
  let gz0: Vec4[T] = Vec4[T](0.5) - abs(gx0) - abs(gy0);
  let sz0: Vec4[T] = step(gz0, Vec4[T](0.0));
  gx0 -= sz0 * (step(T(0), gx0) - T(0.5));
  gy0 -= sz0 * (step(T(0), gy0) - T(0.5));

  let gx1: Vec4[T] = ixy1 * T(1.0 / 7.0);
  let gy1: Vec4[T] = fract(floor(gx1) * T(1.0 / 7.0)) - T(0.5);
  gx1 = fract(gx1);
  let gz1: Vec4[T] = Vec4[T](0.5) - abs(gx1) - abs(gy1);
  let sz1: Vec4[T] = step(gz1, Vec4[T](0.0));
  gx1 -= sz1 * (step(T(0), gx1) - T(0.5));
  gy1 -= sz1 * (step(T(0), gy1) - T(0.5));


  var g000 = vec3(gx0.x, gy0.x, gz0.x);
  var g100 = vec3(gx0.y, gy0.y, gz0.y);
  var g010 = vec3(gx0.z, gy0.z, gz0.z);
  var g110 = vec3(gx0.w, gy0.w, gz0.w);
  var g001 = vec3(gx1.x, gy1.x, gz1.x);
  var g101 = vec3(gx1.y, gy1.y, gz1.y);
  var g011 = vec3(gx1.z, gy1.z, gz1.z);
  var g111 = vec3(gx1.w, gy1.w, gz1.w);

  let norm0 = detail.taylorInvSqrt(Vec4[T](dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
  g000 *= norm0.x;
  g010 *= norm0.y;
  g100 *= norm0.z;
  g110 *= norm0.w;
  let norm1 = detail.taylorInvSqrt(Vec4[T](dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
  g001 *= norm1.x;
  g011 *= norm1.y;
  g101 *= norm1.z;
  g111 *= norm1.w;

  let n000: T = dot(g000, Pf0);
  let n100: T = dot(g100, Vec3[T](Pf1.x, Pf0.y, Pf0.z));
  let n010: T = dot(g010, Vec3[T](Pf0.x, Pf1.y, Pf0.z));
  let n110: T = dot(g110, Vec3[T](Pf1.x, Pf1.y, Pf0.z));
  let n001: T = dot(g001, Vec3[T](Pf0.x, Pf0.y, Pf1.z));
  let n101: T = dot(g101, Vec3[T](Pf1.x, Pf0.y, Pf1.z));
  let n011: T = dot(g011, Vec3[T](Pf0.x, Pf1.y, Pf1.z));
  let n111: T = dot(g111, Pf1);

  let fade_xyz = detail.fade(Pf0);
  let n_z: Vec4[T] = mix(Vec4[T](n000, n100, n010, n110), Vec4[T](n001, n101, n011, n111), fade_xyz.z);
  let n_yz: Vec2[T] = mix(Vec2[T](n_z.x, n_z.y), Vec2[T](n_z.z, n_z.w), fade_xyz.y);
  let n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
  return T(2.2) * n_xyz;

proc perlin*[T](Position: Vec4[T]): T =
  ## Classic Perlin noise

  let Pi0: Vec4[T] = floor(Position);  # Integer part for indexing
  let Pi1: Vec4[T] = Pi0 + T(1);    # Integer part + 1
  Pi0 = mod(Pi0, Vec4[T](289));
  Pi1 = mod(Pi1, Vec4[T](289));
  let Pf0: Vec4[T] = fract(Position);  # Fractional part for interpolation
  let Pf1: Vec4[T] = Pf0 - T(1);    # Fractional part - 1.0
  let ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  let iy = vec4(Pi0.y, Pi0.y, Pi1.y, Pi1.y);
  let iz0 = vec4(Pi0.z);
  let iz1 = vec4(Pi1.z);
  let iw0 = vec4(Pi0.w);
  let iw1 = vec4(Pi1.w);

  let ixy: Vec4[T] = detail.permute(detail.permute(ix) + iy);
  let ixy0: Vec4[T] = detail.permute(ixy + iz0);
  let ixy1: Vec4[T] = detail.permute(ixy + iz1);
  let ixy00: Vec4[T] = detail.permute(ixy0 + iw0);
  let ixy01: Vec4[T] = detail.permute(ixy0 + iw1);
  let ixy10: Vec4[T] = detail.permute(ixy1 + iw0);
  let ixy11: Vec4[T] = detail.permute(ixy1 + iw1);

  let gx00: Vec4[T] = ixy00 / T(7);
  let gy00: Vec4[T] = floor(gx00) / T(7);
  let gz00: Vec4[T] = floor(gy00) / T(6);
  gx00 = fract(gx00) - T(0.5);
  gy00 = fract(gy00) - T(0.5);
  gz00 = fract(gz00) - T(0.5);
  let gw00: Vec4[T] = Vec4[T](0.75) - abs(gx00) - abs(gy00) - abs(gz00);
  let sw00: Vec4[T] = step(gw00, Vec4[T](0.0));
  gx00 -= sw00 * (step(T(0), gx00) - T(0.5));
  gy00 -= sw00 * (step(T(0), gy00) - T(0.5));

  let gx01: Vec4[T] = ixy01 / T(7);
  let gy01: Vec4[T] = floor(gx01) / T(7);
  let gz01: Vec4[T] = floor(gy01) / T(6);
  gx01 = fract(gx01) - T(0.5);
  gy01 = fract(gy01) - T(0.5);
  gz01 = fract(gz01) - T(0.5);
  let gw01: Vec4[T] = Vec4[T](0.75) - abs(gx01) - abs(gy01) - abs(gz01);
  let sw01: Vec4[T] = step(gw01, Vec4[T](0.0));
  gx01 -= sw01 * (step(T(0), gx01) - T(0.5));
  gy01 -= sw01 * (step(T(0), gy01) - T(0.5));

  let gx10: Vec4[T] = ixy10 / T(7);
  let gy10: Vec4[T] = floor(gx10) / T(7);
  let gz10: Vec4[T] = floor(gy10) / T(6);
  gx10 = fract(gx10) - T(0.5);
  gy10 = fract(gy10) - T(0.5);
  gz10 = fract(gz10) - T(0.5);
  let gw10: Vec4[T] = Vec4[T](0.75) - abs(gx10) - abs(gy10) - abs(gz10);
  let sw10: Vec4[T] = step(gw10, Vec4[T](0));
  gx10 -= sw10 * (step(T(0), gx10) - T(0.5));
  gy10 -= sw10 * (step(T(0), gy10) - T(0.5));

  let gx11: Vec4[T] = ixy11 / T(7);
  let gy11: Vec4[T] = floor(gx11) / T(7);
  let gz11: Vec4[T] = floor(gy11) / T(6);
  gx11 = fract(gx11) - T(0.5);
  gy11 = fract(gy11) - T(0.5);
  gz11 = fract(gz11) - T(0.5);
  let gw11: Vec4[T] = Vec4[T](0.75) - abs(gx11) - abs(gy11) - abs(gz11);
  let sw11: Vec4[T] = step(gw11, Vec4[T](0.0));
  gx11 -= sw11 * (step(T(0), gx11) - T(0.5));
  gy11 -= sw11 * (step(T(0), gy11) - T(0.5));

  var
    g0000 = vec4(gx00.x, gy00.x, gz00.x, gw00.x)
    g1000 = vec4(gx00.y, gy00.y, gz00.y, gw00.y)
    g0100 = vec4(gx00.z, gy00.z, gz00.z, gw00.z)
    g1100 = vec4(gx00.w, gy00.w, gz00.w, gw00.w)
    g0010 = vec4(gx10.x, gy10.x, gz10.x, gw10.x)
    g1010 = vec4(gx10.y, gy10.y, gz10.y, gw10.y)
    g0110 = vec4(gx10.z, gy10.z, gz10.z, gw10.z)
    g1110 = vec4(gx10.w, gy10.w, gz10.w, gw10.w)
    g0001 = vec4(gx01.x, gy01.x, gz01.x, gw01.x)
    g1001 = vec4(gx01.y, gy01.y, gz01.y, gw01.y)
    g0101 = vec4(gx01.z, gy01.z, gz01.z, gw01.z)
    g1101 = vec4(gx01.w, gy01.w, gz01.w, gw01.w)
    g0011 = vec4(gx11.x, gy11.x, gz11.x, gw11.x)
    g1011 = vec4(gx11.y, gy11.y, gz11.y, gw11.y)
    g0111 = vec4(gx11.z, gy11.z, gz11.z, gw11.z)
    g1111 = vec4(gx11.w, gy11.w, gz11.w, gw11.w)

  let norm00: Vec4[T] = detail.taylorInvSqrt(Vec4[T](dot(g0000, g0000), dot(g0100, g0100), dot(g1000, g1000), dot(g1100, g1100)));
  g0000 *= norm00.x;
  g0100 *= norm00.y;
  g1000 *= norm00.z;
  g1100 *= norm00.w;

  let norm01: Vec4[T] = detail.taylorInvSqrt(Vec4[T](dot(g0001, g0001), dot(g0101, g0101), dot(g1001, g1001), dot(g1101, g1101)));
  g0001 *= norm01.x;
  g0101 *= norm01.y;
  g1001 *= norm01.z;
  g1101 *= norm01.w;

  let norm10: Vec4[T] = detail.taylorInvSqrt(Vec4[T](dot(g0010, g0010), dot(g0110, g0110), dot(g1010, g1010), dot(g1110, g1110)));
  g0010 *= norm10.x;
  g0110 *= norm10.y;
  g1010 *= norm10.z;
  g1110 *= norm10.w;

  let norm11: Vec4[T] = detail.taylorInvSqrt(Vec4[T](dot(g0011, g0011), dot(g0111, g0111), dot(g1011, g1011), dot(g1111, g1111)));
  g0011 *= norm11.x;
  g0111 *= norm11.y;
  g1011 *= norm11.z;
  g1111 *= norm11.w;

  let n0000: T = dot(g0000, Pf0);
  let n1000: T = dot(g1000, Vec4[T](Pf1.x, Pf0.y, Pf0.z, Pf0.w));
  let n0100: T = dot(g0100, Vec4[T](Pf0.x, Pf1.y, Pf0.z, Pf0.w));
  let n1100: T = dot(g1100, Vec4[T](Pf1.x, Pf1.y, Pf0.z, Pf0.w));
  let n0010: T = dot(g0010, Vec4[T](Pf0.x, Pf0.y, Pf1.z, Pf0.w));
  let n1010: T = dot(g1010, Vec4[T](Pf1.x, Pf0.y, Pf1.z, Pf0.w));
  let n0110: T = dot(g0110, Vec4[T](Pf0.x, Pf1.y, Pf1.z, Pf0.w));
  let n1110: T = dot(g1110, Vec4[T](Pf1.x, Pf1.y, Pf1.z, Pf0.w));
  let n0001: T = dot(g0001, Vec4[T](Pf0.x, Pf0.y, Pf0.z, Pf1.w));
  let n1001: T = dot(g1001, Vec4[T](Pf1.x, Pf0.y, Pf0.z, Pf1.w));
  let n0101: T = dot(g0101, Vec4[T](Pf0.x, Pf1.y, Pf0.z, Pf1.w));
  let n1101: T = dot(g1101, Vec4[T](Pf1.x, Pf1.y, Pf0.z, Pf1.w));
  let n0011: T = dot(g0011, Vec4[T](Pf0.x, Pf0.y, Pf1.z, Pf1.w));
  let n1011: T = dot(g1011, Vec4[T](Pf1.x, Pf0.y, Pf1.z, Pf1.w));
  let n0111: T = dot(g0111, Vec4[T](Pf0.x, Pf1.y, Pf1.z, Pf1.w));
  let n1111: T = dot(g1111, Pf1);

  let fade_xyzw: Vec4[T] = detail.fade(Pf0);
  let n_0w: Vec4[T] = mix(Vec4[T](n0000, n1000, n0100, n1100), Vec4[T](n0001, n1001, n0101, n1101), fade_xyzw.w);
  let n_1w: Vec4[T] = mix(Vec4[T](n0010, n1010, n0110, n1110), Vec4[T](n0011, n1011, n0111, n1111), fade_xyzw.w);
  let n_zw: Vec4[T] = mix(n_0w, n_1w, fade_xyzw.z);
  let n_yzw: Vec2[T] = mix(Vec2[T](n_zw.x, n_zw.y), Vec2[T](n_zw.z, n_zw.w), fade_xyzw.y);
  let n_xyzw: T = mix(n_yzw.x, n_yzw.y, fade_xyzw.x);
  return T(2.2) * n_xyzw;


# Classic Perlin noise, periodic variant
proc perlin*[T](Position: Vec2[T]; rep: Vec2[T]): T =

  let Pi: Vec4[T] = floor(Vec4[T](Position.x, Position.y, Position.x, Position.y)) + Vec4[T](0.0, 0.0, 1.0, 1.0);
  let Pf: Vec4[T] = fract(Vec4[T](Position.x, Position.y, Position.x, Position.y)) - Vec4[T](0.0, 0.0, 1.0, 1.0);
  Pi = mod(Pi, Vec4[T](rep.x, rep.y, rep.x, rep.y)); # To create noise with explicit period
  Pi = mod(Pi, Vec4[T](289)); # To avoid truncation effects in permutation
  let ix = vec4(Pi.x, Pi.z, Pi.x, Pi.z);
  let iy = vec4(Pi.y, Pi.y, Pi.w, Pi.w);
  let fx = vec4(Pf.x, Pf.z, Pf.x, Pf.z);
  let fy = vec4(Pf.y, Pf.y, Pf.w, Pf.w);

  let i: Vec4[T] = detail.permute(detail.permute(ix) + iy);

  let gx: Vec4[T] = T(2) * fract(i / T(41)) - T(1);
  let gy: Vec4[T] = abs(gx) - T(0.5);
  let tx: Vec4[T] = floor(gx + T(0.5));
  gx = gx - tx;

  let g00 = vec2(gx.x, gy.x);
  let g10 = vec2(gx.y, gy.y);
  let g01 = vec2(gx.z, gy.z);
  let g11 = vec2(gx.w, gy.w);

  let norm: Vec4[T] = detail.taylorInvSqrt(Vec4[T](dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
  g00 *= norm.x;
  g01 *= norm.y;
  g10 *= norm.z;
  g11 *= norm.w;

  let n00: T = dot(g00, Vec2[T](fx.x, fy.x));
  let n10: T = dot(g10, Vec2[T](fx.y, fy.y));
  let n01: T = dot(g01, Vec2[T](fx.z, fy.z));
  let n11: T = dot(g11, Vec2[T](fx.w, fy.w));

  let fade_xy: Vec2[T] = detail.fade(Vec2[T](Pf.x, Pf.y));
  let n_x: Vec2[T] = mix(Vec2[T](n00, n01), Vec2[T](n10, n11), fade_xy.x);
  let n_xy: T = mix(n_x.x, n_x.y, fade_xy.y);
  return T(2.3) * n_xy;


proc perlin*[T](Position: Vec3[T]; rep: Vec3[T]): T =
  ## Classic Perlin noise, periodic variant

  let Pi0: Vec3[T] = mod(floor(Position), rep); # Integer part, modulo period
  let Pi1: Vec3[T] = mod(Pi0 + Vec3[T](T(1)), rep); # Integer part + 1, mod period
  Pi0 = mod(Pi0, Vec3[T](289));
  Pi1 = mod(Pi1, Vec3[T](289));
  let Pf0: Vec3[T] = fract(Position); # Fractional part for interpolation
  let Pf1: Vec3[T] = Pf0 - Vec3[T](T(1)); # Fractional part - 1.0
  let ix: Vec4[T] = Vec4[T](Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  let iy: Vec4[T] = Vec4[T](Pi0.y, Pi0.y, Pi1.y, Pi1.y);
  let iz0 = vec4(Pi0.z);
  let iz1 = vec4(Pi1.z);

  let ixy: Vec4[T] = detail.permute(detail.permute(ix) + iy);
  let ixy0: Vec4[T] = detail.permute(ixy + iz0);
  let ixy1: Vec4[T] = detail.permute(ixy + iz1);

  let gx0: Vec4[T] = ixy0 / T(7);
  let gy0: Vec4[T] = fract(floor(gx0) / T(7)) - T(0.5);
  gx0 = fract(gx0);
  let gz0: Vec4[T] = Vec4[T](0.5) - abs(gx0) - abs(gy0);
  let sz0: Vec4[T] = step(gz0, Vec4[T](0));
  gx0 -= sz0 * (step(T(0), gx0) - T(0.5));
  gy0 -= sz0 * (step(T(0), gy0) - T(0.5));

  let gx1: Vec4[T] = ixy1 / T(7);
  let gy1: Vec4[T] = fract(floor(gx1) / T(7)) - T(0.5);
  gx1 = fract(gx1);
  let gz1: Vec4[T] = Vec4[T](0.5) - abs(gx1) - abs(gy1);
  let sz1: Vec4[T] = step(gz1, Vec4[T](T(0)));
  gx1 -= sz1 * (step(T(0), gx1) - T(0.5));
  gy1 -= sz1 * (step(T(0), gy1) - T(0.5));

  let g000: Vec3[T] = Vec3[T](gx0.x, gy0.x, gz0.x);
  let g100: Vec3[T] = Vec3[T](gx0.y, gy0.y, gz0.y);
  let g010: Vec3[T] = Vec3[T](gx0.z, gy0.z, gz0.z);
  let g110: Vec3[T] = Vec3[T](gx0.w, gy0.w, gz0.w);
  let g001: Vec3[T] = Vec3[T](gx1.x, gy1.x, gz1.x);
  let g101: Vec3[T] = Vec3[T](gx1.y, gy1.y, gz1.y);
  let g011: Vec3[T] = Vec3[T](gx1.z, gy1.z, gz1.z);
  let g111: Vec3[T] = Vec3[T](gx1.w, gy1.w, gz1.w);

  let norm0: Vec4[T] = detail.taylorInvSqrt(Vec4[T](dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
  g000 *= norm0.x;
  g010 *= norm0.y;
  g100 *= norm0.z;
  g110 *= norm0.w;
  let norm1: Vec4[T] = detail.taylorInvSqrt(Vec4[T](dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
  g001 *= norm1.x;
  g011 *= norm1.y;
  g101 *= norm1.z;
  g111 *= norm1.w;

  let n000 = dot(g000, Pf0);
  let n100 = dot(g100, Vec3[T](Pf1.x, Pf0.y, Pf0.z));
  let n010 = dot(g010, Vec3[T](Pf0.x, Pf1.y, Pf0.z));
  let n110 = dot(g110, Vec3[T](Pf1.x, Pf1.y, Pf0.z));
  let n001 = dot(g001, Vec3[T](Pf0.x, Pf0.y, Pf1.z));
  let n101 = dot(g101, Vec3[T](Pf1.x, Pf0.y, Pf1.z));
  let n011 = dot(g011, Vec3[T](Pf0.x, Pf1.y, Pf1.z));
  let n111 = dot(g111, Pf1);

  let fade_xyz: Vec3[T] = detail.fade(Pf0);
  let n_z: Vec4[T] = mix(Vec4[T](n000, n100, n010, n110), Vec4[T](n001, n101, n011, n111), fade_xyz.z);
  let n_yz: Vec2[T] = mix(Vec2[T](n_z.x, n_z.y), Vec2[T](n_z.z, n_z.w), fade_xyz.y);
  let n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
  return T(2.2) * n_xyz;

proc perlin*[T](Position: Vec4[T]; rep: Vec4[T]): T =
  ## Classic Perlin noise, periodic version

  let Pi0: Vec4[T] = mod(floor(Position), rep); # Integer part modulo rep
  let Pi1: Vec4[T] = mod(Pi0 + T(1), rep); # Integer part + 1 mod rep
  let Pf0: Vec4[T] = fract(Position); # Fractional part for interpolation
  let Pf1: Vec4[T] = Pf0 - T(1); # Fractional part - 1.0
  let ix: Vec4[T] = Vec4[T](Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  let iy: Vec4[T] = Vec4[T](Pi0.y, Pi0.y, Pi1.y, Pi1.y);
  let iz0 = vec4(Pi0.z);
  let iz1 = vec4(Pi1.z);
  let iw0 = vec4(Pi0.w);
  let iw1 = vec4(Pi1.w);

  let ixy: Vec4[T] = detail.permute(detail.permute(ix) + iy);
  let ixy0: Vec4[T] = detail.permute(ixy + iz0);
  let ixy1: Vec4[T] = detail.permute(ixy + iz1);
  let ixy00: Vec4[T] = detail.permute(ixy0 + iw0);
  let ixy01: Vec4[T] = detail.permute(ixy0 + iw1);
  let ixy10: Vec4[T] = detail.permute(ixy1 + iw0);
  let ixy11: Vec4[T] = detail.permute(ixy1 + iw1);

  let gx00: Vec4[T] = ixy00 / T(7);
  let gy00: Vec4[T] = floor(gx00) / T(7);
  let gz00: Vec4[T] = floor(gy00) / T(6);
  gx00 = fract(gx00) - T(0.5);
  gy00 = fract(gy00) - T(0.5);
  gz00 = fract(gz00) - T(0.5);
  let gw00: Vec4[T] = Vec4[T](0.75) - abs(gx00) - abs(gy00) - abs(gz00);
  let sw00: Vec4[T] = step(gw00, Vec4[T](0));
  gx00 -= sw00 * (step(T(0), gx00) - T(0.5));
  gy00 -= sw00 * (step(T(0), gy00) - T(0.5));

  let gx01: Vec4[T] = ixy01 / T(7);
  let gy01: Vec4[T] = floor(gx01) / T(7);
  let gz01: Vec4[T] = floor(gy01) / T(6);
  gx01 = fract(gx01) - T(0.5);
  gy01 = fract(gy01) - T(0.5);
  gz01 = fract(gz01) - T(0.5);
  let gw01: Vec4[T] = Vec4[T](0.75) - abs(gx01) - abs(gy01) - abs(gz01);
  let sw01: Vec4[T] = step(gw01, Vec4[T](0.0));
  gx01 -= sw01 * (step(T(0), gx01) - T(0.5));
  gy01 -= sw01 * (step(T(0), gy01) - T(0.5));

  let gx10: Vec4[T] = ixy10 / T(7);
  let gy10: Vec4[T] = floor(gx10) / T(7);
  let gz10: Vec4[T] = floor(gy10) / T(6);
  gx10 = fract(gx10) - T(0.5);
  gy10 = fract(gy10) - T(0.5);
  gz10 = fract(gz10) - T(0.5);
  let gw10: Vec4[T] = Vec4[T](0.75) - abs(gx10) - abs(gy10) - abs(gz10);
  let sw10: Vec4[T] = step(gw10, Vec4[T](0.0));
  gx10 -= sw10 * (step(T(0), gx10) - T(0.5));
  gy10 -= sw10 * (step(T(0), gy10) - T(0.5));

  let gx11: Vec4[T] = ixy11 / T(7);
  let gy11: Vec4[T] = floor(gx11) / T(7);
  let gz11: Vec4[T] = floor(gy11) / T(6);
  gx11 = fract(gx11) - T(0.5);
  gy11 = fract(gy11) - T(0.5);
  gz11 = fract(gz11) - T(0.5);
  let gw11: Vec4[T] = Vec4[T](0.75) - abs(gx11) - abs(gy11) - abs(gz11);
  let sw11: Vec4[T] = step(gw11, Vec4[T](T(0)));
  gx11 -= sw11 * (step(T(0), gx11) - T(0.5));
  gy11 -= sw11 * (step(T(0), gy11) - T(0.5));

  var g0000 = vec4(gx00.x, gy00.x, gz00.x, gw00.x);
  var g1000 = vec4(gx00.y, gy00.y, gz00.y, gw00.y);
  var g0100 = vec4(gx00.z, gy00.z, gz00.z, gw00.z);
  var g1100 = vec4(gx00.w, gy00.w, gz00.w, gw00.w);
  var g0010 = vec4(gx10.x, gy10.x, gz10.x, gw10.x);
  var g1010 = vec4(gx10.y, gy10.y, gz10.y, gw10.y);
  var g0110 = vec4(gx10.z, gy10.z, gz10.z, gw10.z);
  var g1110 = vec4(gx10.w, gy10.w, gz10.w, gw10.w);
  var g0001 = vec4(gx01.x, gy01.x, gz01.x, gw01.x);
  var g1001 = vec4(gx01.y, gy01.y, gz01.y, gw01.y);
  var g0101 = vec4(gx01.z, gy01.z, gz01.z, gw01.z);
  var g1101 = vec4(gx01.w, gy01.w, gz01.w, gw01.w);
  var g0011 = vec4(gx11.x, gy11.x, gz11.x, gw11.x);
  var g1011 = vec4(gx11.y, gy11.y, gz11.y, gw11.y);
  var g0111 = vec4(gx11.z, gy11.z, gz11.z, gw11.z);
  var g1111 = vec4(gx11.w, gy11.w, gz11.w, gw11.w);

  let norm00 = detail.taylorInvSqrt(Vec4[T](dot(g0000, g0000), dot(g0100, g0100), dot(g1000, g1000), dot(g1100, g1100)));
  g0000 *= norm00.x;
  g0100 *= norm00.y;
  g1000 *= norm00.z;
  g1100 *= norm00.w;

  let norm01 = detail.taylorInvSqrt(Vec4[T](dot(g0001, g0001), dot(g0101, g0101), dot(g1001, g1001), dot(g1101, g1101)));
  g0001 *= norm01.x;
  g0101 *= norm01.y;
  g1001 *= norm01.z;
  g1101 *= norm01.w;

  let norm10 = detail.taylorInvSqrt(Vec4[T](dot(g0010, g0010), dot(g0110, g0110), dot(g1010, g1010), dot(g1110, g1110)));
  g0010 *= norm10.x;
  g0110 *= norm10.y;
  g1010 *= norm10.z;
  g1110 *= norm10.w;

  let norm11 = detail.taylorInvSqrt(Vec4[T](dot(g0011, g0011), dot(g0111, g0111), dot(g1011, g1011), dot(g1111, g1111)));
  g0011 *= norm11.x;
  g0111 *= norm11.y;
  g1011 *= norm11.z;
  g1111 *= norm11.w;

  let n0000: T = dot(g0000, Pf0);
  let n1000: T = dot(g1000, Vec4[T](Pf1.x, Pf0.y, Pf0.z, Pf0.w));
  let n0100: T = dot(g0100, Vec4[T](Pf0.x, Pf1.y, Pf0.z, Pf0.w));
  let n1100: T = dot(g1100, Vec4[T](Pf1.x, Pf1.y, Pf0.z, Pf0.w));
  let n0010: T = dot(g0010, Vec4[T](Pf0.x, Pf0.y, Pf1.z, Pf0.w));
  let n1010: T = dot(g1010, Vec4[T](Pf1.x, Pf0.y, Pf1.z, Pf0.w));
  let n0110: T = dot(g0110, Vec4[T](Pf0.x, Pf1.y, Pf1.z, Pf0.w));
  let n1110: T = dot(g1110, Vec4[T](Pf1.x, Pf1.y, Pf1.z, Pf0.w));
  let n0001: T = dot(g0001, Vec4[T](Pf0.x, Pf0.y, Pf0.z, Pf1.w));
  let n1001: T = dot(g1001, Vec4[T](Pf1.x, Pf0.y, Pf0.z, Pf1.w));
  let n0101: T = dot(g0101, Vec4[T](Pf0.x, Pf1.y, Pf0.z, Pf1.w));
  let n1101: T = dot(g1101, Vec4[T](Pf1.x, Pf1.y, Pf0.z, Pf1.w));
  let n0011: T = dot(g0011, Vec4[T](Pf0.x, Pf0.y, Pf1.z, Pf1.w));
  let n1011: T = dot(g1011, Vec4[T](Pf1.x, Pf0.y, Pf1.z, Pf1.w));
  let n0111: T = dot(g0111, Vec4[T](Pf0.x, Pf1.y, Pf1.z, Pf1.w));
  let n1111: T = dot(g1111, Pf1);

  let fade_xyzw: Vec4[T] = detail.fade(Pf0);
  let n_0w: Vec4[T] = mix(Vec4[T](n0000, n1000, n0100, n1100), Vec4[T](n0001, n1001, n0101, n1101), fade_xyzw.w);
  let n_1w: Vec4[T] = mix(Vec4[T](n0010, n1010, n0110, n1110), Vec4[T](n0011, n1011, n0111, n1111), fade_xyzw.w);
  let n_zw: Vec4[T] = mix(n_0w, n_1w, fade_xyzw.z);
  let n_yzw: Vec2[T] = mix(Vec2[T](n_zw.x, n_zw.y), Vec2[T](n_zw.z, n_zw.w), fade_xyzw.y);
  let n_xyzw: T = mix(n_yzw.x, n_yzw.y, fade_xyzw.x);
  return T(2.2) * n_xyzw;


proc simplex*[T](v: Vec2[T]): T =

  const
    C = Vec4[T](
      T( 0.211324865405187),  # (3.0 -  sqrt(3.0)) / 6.0
      T( 0.366025403784439),  #  0.5 * (sqrt(3.0)  - 1.0)
      T(-0.577350269189626),  # -1.0 + 2.0 * C.x
      T( 0.024390243902439))  #  1.0 / 41.0

  # First corner
  var i: Vec2[T]  = floor(v + dot(v, Vec2[T](C[1])));
  let x0: Vec2[T] = v -   i + dot(i, Vec2[T](C[0]));

  # Other corners
  #i1.x = step( x0.y, x0.x ); # x0.x ] x0.y ? 1.0 : 0.0
  #i1.y = 1.0 - i1.x;
  let i1: Vec2[T] = if x0.x > x0.y: vec2[T](1, 0) else: vec2[T](0, 1)
  # x0 = x0 - 0.0 + 0.0 * C.xx ;
  # x1 = x0 - i1 + 1.0 * C.xx ;
  # x2 = x0 - 1.0 + 2.0 * C.xx ;
  let x12: Vec4[T] = vec4[T](x0.x, x0.y, x0.x, x0.y) + vec4[T](C.x, C.x, C.z, C.z);
  x12 = Vec4[T](Vec2[T](x12) - i1, x12.z, x12.w);

  # Permutations
  i = mod(i, vec2[T](289)); # Avoid truncation effects in permutation
  let p: Vec3[T] = detail.permute(
    detail.permute(i.y + vec3[T](T(0), i1.y, T(1))) +
    i.x + vec3[T](T(0), i1.x, T(1)));

  let m: Vec3[T] = max(vec3[T](0.5) - Vec3[T](
    dot(x0, x0),
    dot(Vec2[T](x12.x, x12.y), Vec2[T](x12.x, x12.y)),
    dot(Vec2[T](x12.z, x12.w), Vec2[T](x12.z, x12.w))), Vec3[T](0));
  m = m * m ;
  m = m * m ;

  # Gradients: 41 points uniformly over a line, mapped onto a diamond.
  # The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

  let x: Vec3[T] = T(2) * fract(p * C.w) - T(1);
  let h: Vec3[T] = abs(x) - T(0.5);
  let ox: Vec3[T] = floor(x + T(0.5));
  let a0: Vec3[T] = x - ox;

  # Normalise gradients implicitly by scaling m
  # Inlined for speed: m *= taylorInvSqrt( a0*a0 + h*h );
  m *= T(1.79284291400159) - T(0.85373472095314) * (a0 * a0 + h * h);

  # Compute final noise value at P
  Vec3[T] g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  #g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  g.y = a0.y * x12.x + h.y * x12.y;
  g.z = a0.z * x12.z + h.z * x12.w;
  return T(130) * dot(m, g);

proc simplex*[T](v: Vec3[T]): T =

  const
    C = vec2[T](1.0 / 6.0, 1.0 / 3.0)
    D = vec4[T](0.0, 0.5, 1.0, 2.0);

  # First corner
  let i = vec3[T](floor(v + dot(v, Vec3[T](C.y))));
  let x0 = vec3(v - i + dot(i, Vec3[T](C.x)));

  # Other corners
  let g = vec3[T](step(Vec3[T](x0.y, x0.z, x0.x), x0));
  let l = vec3[T](T(1) - g);
  let i1 = vec3[T](min(g, Vec3[T](l.z, l.x, l.y)));
  let i2 = vec3[T](max(g, Vec3[T](l.z, l.x, l.y)));

  #   x0 = x0 - 0.0 + 0.0 * C.xxx;
  #   x1 = x0 - i1  + 1.0 * C.xxx;
  #   x2 = x0 - i2  + 2.0 * C.xxx;
  #   x3 = x0 - 1.0 + 3.0 * C.xxx;
  let x1 = vec3(x0 - i1 + C.x);
  let x2 = vec3(x0 - i2 + C.y); # 2.0*C.x = 1/3 = C.y
  let x3 = vec3(x0 - D.y);      # -1.0+3.0*C.x = -0.5 = -D.y

  # Permutations
  i = detail.mod289(i);
  Vec4[T] p(detail.permute(detail.permute(detail.permute(
    i.z + Vec4[T](T(0), i1.z, i2.z, T(1))) +
    i.y + Vec4[T](T(0), i1.y, i2.y, T(1))) +
    i.x + Vec4[T](T(0), i1.x, i2.x, T(1))));

  # Gradients: 7x7 points over a square, mapped onto an octahedron.
  # The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  let n = T(0.142857142857); # 1.0/7.0
  let ns = vec3[T](n * Vec3[T](D.w, D.y, D.z) - Vec3[T](D.x, D.z, D.x));

  let j = vec4[T](p - T(49) * floor(p * ns.z * ns.z));  #  mod(p,7*7)

  let x_u = vec4[T](floor(j * ns.z));
  let y_u = vec4[T](floor(j - T(7) * x_u));    # mod(j,N)

  let x = vec4[T](x_u * ns.x + ns.y);
  let y = vec4[T](y_u * ns.x + ns.y);
  let h = vec4[T](T(1) - abs(x) - abs(y));

  let b0 = vec4[T](x.x, x.y, y.x, y.y);
  let b1 = vec4[T](x.z, x.w, y.z, y.w);

  # vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  # vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  let s0 = vec4[T](floor(b0) * T(2) + T(1));
  let s1 = vec4[T](floor(b1) * T(2) + T(1));
  let sh = vec4[T](-step(h, Vec4[T](0.0)));

  let a0: Vec4[T] = Vec4[T](b0.x, b0.z, b0.y, b0.w) + Vec4[T](s0.x, s0.z, s0.y, s0.w) * Vec4[T](sh.x, sh.x, sh.y, sh.y);
  let a1: Vec4[T] = Vec4[T](b1.x, b1.z, b1.y, b1.w) + Vec4[T](s1.x, s1.z, s1.y, s1.w) * Vec4[T](sh.z, sh.z, sh.w, sh.w);

  let p0 = vec3[T](a0.x, a0.y, h.x);
  let p1 = vec3[T](a0.z, a0.w, h.y);
  let p2 = vec3[T](a1.x, a1.y, h.z);
  let p3 = vec3[T](a1.z, a1.w, h.w);

  # Normalise gradients
  let norm: Vec4[T] = detail.taylorInvSqrt(Vec4[T](dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

  # Mix final noise value
  let m: Vec4[T] = max(T(0.6) - Vec4[T](dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), Vec4[T](0));
  m = m * m;
  return T(42) * dot(m * m, Vec4[T](dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));


proc simplex*[T](v: Vec4[T]): T =

  const
    C = vec4[T](
      0.138196601125011,  # (5 - sqrt(5))/20  G4
      0.276393202250021,  # 2 * G4
      0.414589803375032,  # 3 * G4
      -0.447213595499958  # -1 + 4 * G4
    )

    # (sqrt(5) - 1)/4 = F4, used once below
    F4 = T(0.309016994374947451);

  # First corner
  let i: Vec4[T]  = floor(v + dot(v, vec4(F4)));
  let x0: Vec4[T] = v -   i + dot(i, vec4(C.x));

  # Other corners

  # Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
  var i0: Vec4[T];
  let isX: Vec3[T] = step(Vec3[T](x0.y, x0.z, x0.w), Vec3[T](x0.x));
  let isYZ: Vec3[T] = step(Vec3[T](x0.z, x0.w, x0.w), Vec3[T](x0.y, x0.y, x0.z));
  #  i0.x = dot(isX, vec3(1.0));
  #i0.x = isX.x + isX.y + isX.z;
  #i0.yzw = T(1) - isX;
  i0 = Vec4[T](isX.x + isX.y + isX.z, T(1) - isX);
  #  i0.y += dot(isYZ.xy, vec2(1.0));
  i0.y += isYZ.x + isYZ.y;
  #i0.zw += 1.0 - Vec2[T](isYZ.x, isYZ.y);
  i0.z += T(1) - isYZ.x;
  i0.w += T(1) - isYZ.y;
  i0.z += isYZ.z;
  i0.w += T(1) - isYZ.z;

  # i0 now contains the unique values 0,1,2,3 in each channel
  let i3: Vec4[T] = clamp(i0, T(0), T(1));
  let i2: Vec4[T] = clamp(i0 - T(1), T(0), T(1));
  let i1: Vec4[T] = clamp(i0 - T(2), T(0), T(1));

  #  x0 = x0 - 0.0 + 0.0 * C.xxxx
  #  x1 = x0 - i1  + 0.0 * C.xxxx
  #  x2 = x0 - i2  + 0.0 * C.xxxx
  #  x3 = x0 - i3  + 0.0 * C.xxxx
  #  x4 = x0 - 1.0 + 4.0 * C.xxxx
  let x1: Vec4[T] = x0 - i1 + C.x;
  let x2: Vec4[T] = x0 - i2 + C.y;
  let x3: Vec4[T] = x0 - i3 + C.z;
  let x4: Vec4[T] = x0 + C.w;

  # Permutations
  i = mod(i, Vec4[T](289));
  let j0: T = detail.permute(detail.permute(detail.permute(detail.permute(i.w) + i.z) + i.y) + i.x);
  let j1: Vec4[T] = detail.permute(detail.permute(detail.permute(detail.permute(
    i.w + Vec4[T](i1.w, i2.w, i3.w, T(1))) +
    i.z + Vec4[T](i1.z, i2.z, i3.z, T(1))) +
    i.y + Vec4[T](i1.y, i2.y, i3.y, T(1))) +
    i.x + Vec4[T](i1.x, i2.x, i3.x, T(1)));

  # Gradients: 7x7x6 points over a cube, mapped onto a 4-cross polytope
  # 7*7*6 = 294, which is close to the ring size 17*17 = 289.
  let ip: Vec4[T] = Vec4[T](T(1) / T(294), T(1) / T(49), T(1) / T(7), T(0));

  let p0: Vec4[T] = gtc_grad4(j0,   ip);
  let p1: Vec4[T] = gtc_grad4(j1.x, ip);
  let p2: Vec4[T] = gtc_grad4(j1.y, ip);
  let p3: Vec4[T] = gtc_grad4(j1.z, ip);
  let p4: Vec4[T] = gtc_grad4(j1.w, ip);

  # Normalise gradients
  let norm: Vec4[T] = detail.taylorInvSqrt(Vec4[T](dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;
  p4 *= detail.taylorInvSqrt(dot(p4, p4));

  # Mix contributions from the five corners
  let m0: Vec3[T] = max(T(0.6) - Vec3[T](dot(x0, x0), dot(x1, x1), dot(x2, x2)), Vec3[T](0));
  let m1: Vec2[T] = max(T(0.6) - Vec2[T](dot(x3, x3), dot(x4, x4)             ), Vec2[T](0));
  m0 = m0 * m0;
  m1 = m1 * m1;
  return T(49) *
    (dot(m0 * m0, vec3[T](dot(p0, x0), dot(p1, x1), dot(p2, x2))) +
    dot(m1 * m1, vec2[T](dot(p3, x3), dot(p4, x4))));

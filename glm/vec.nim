import strutils
type
    Vec1[T] = object
        a:array[1,T]
    Vec2[T] = object
        a:array[2,T]
    Vec3[T] = object 
        a:array[3,T]
    Vec4[T] = object 
        a:array[4,T]
    Mat2[T] = object 
        a:array[2*2,T]
    Mat3[T] = object 
        a:array[3*3,T]
    Mat4[T] = object 
        a:array[4*4,T]

proc `$`*[T](a:openArray[T]):string=
    var 
        strs:seq[string] = @[]
    for i in a:
        strs.add( $i);
    return [ "[" ,strs.join(", "),"]" ].join

proc `$`*[T](v:Vec1[T]):string= $v.a
proc `$`*[T](v:Vec2[T]):string= $v.a
proc `$`*[T](v:Vec3[T]):string= $v.a
proc `$`*[T](v:Vec4[T]):string= $v.a
proc `$`*[T](v:Mat2[T]):string= $v.a
proc `$`*[T](v:Mat3[T]):string= $v.a
proc `$`*[T](v:Mat4[T]):string= $v.a

proc `[]`*[T](v:Vec1[T], idx:int):T=v.a[idx]
proc `[]`*[T](v:Vec2[T], idx:int):T=v.a[idx]
proc `[]`*[T](v:Vec3[T], idx:int):T=v.a[idx]
proc `[]`*[T](v:Vec4[T], idx:int):T=v.a[idx]
proc `[]`*[T](v:Mat2[T], idx:int):T=v.a[idx]
proc `[]`*[T](v:Mat3[T], idx:int):T=v.a[idx]
proc `[]`*[T](v:Mat4[T], idx:int):T=v.a[idx]

proc `[]=`*[T](v:var Vec1[T], x:int, c:T)=`[]=`(v.a, x..x,[c])
proc `[]=`*[T](v:var Vec2[T], x:int, c:T)=`[]=`(v.a, x..x,[c])
proc `[]=`*[T](v:var Vec3[T], x:int, c:T)=`[]=`(v.a, x..x,[c])
proc `[]=`*[T](v:var Vec4[T], x:int, c:T)=`[]=`(v.a, x..x,[c])
proc `[]=`*[T](v:var Mat2[T], x:int, c:T)=`[]=`(v.a, x..x,[c])
proc `[]=`*[T](v:var Mat3[T], x:int, c:T)=`[]=`(v.a, x..x,[c])
proc `[]=`*[T](v:var Mat4[T], x:int, c:T)=`[]=`(v.a, x..x,[c])

proc addr*[T](v:var Vec1[T]):ptr T= addr v.a[0]
proc addr*[T](v:var Vec2[T]):ptr T= addr v.a[0]
proc addr*[T](v:var Vec3[T]):ptr T= addr v.a[0]
proc addr*[T](v:var Vec4[T]):ptr T= addr v.a[0]
proc addr*[T](v:var Mat2[T]):ptr T= addr v.a[0]
proc addr*[T](v:var Mat3[T]):ptr T= addr v.a[0]
proc addr*[T](v:var Mat4[T]):ptr T= addr v.a[0]

# literal constructors
proc vec1*[T](x:T):Vec1[T]= Vec1[T](a:[x])
proc vec1*():Vec1[float]= Vec1[float](a:[0.float])
proc vec2*[T](x,y:T):Vec2[T]= Vec2[T](a:[x,y])
proc vec2*[T](x:T):Vec2[T]= Vec2[T](a:[x,x])
proc vec2*():Vec2[float]= vec2(0.float)
proc vec3*[T](x,y,z:T):Vec3[T]= Vec3[T](a:[x,y,z])
proc vec3*[T](x:T):Vec3[T]= Vec3[T](a:[x,x,x])
proc vec3*():Vec3[float]= vec3(0.float)
proc vec4*[T](x,y,z,w:T):Vec4[T]= Vec4[T](a:[x,y,z,w])
proc vec4*[T](x:T):Vec4[T]= Vec4[T](a:[x,x,x,x])
proc vec4*():Vec4[float]= vec4(0.float)
# complex constructors
proc vec1*[T](v:Vec1[T]):Vec1[T]= vec1(v[0])
proc vec2*[T](v:Vec2[T]):Vec2[T]= vec2(v[0],v[1])
proc vec3*[T](v:Vec3[T]):Vec3[T]= vec3(v[0],v[1],v[2])
proc vec4*[T](v:Vec4[T]):Vec4[T]= vec4(v[0],v[1],v[2],v[3])

proc vec3*[T](v2:Vec2[T], v:T):Vec3[T]= vec3(v2[0], v2[1], v)
proc vec3*[T](v:T, v2:Vec2[T]):Vec3[T]= vec3(v, v2[0], v2[1])

proc vec4*[T](v:T, u:T, v2:Vec2[T]):Vec4[T]= vec4(v, u, v2[0], v2[1])
proc vec4*[T](v:T, v3:Vec3[T]):Vec4[T]= vec4(v, v3[0], v3[1], v3[2])
proc vec4*[T](v3:Vec3[T], v:T):Vec4[T]= vec4( v3[0], v3[1], v3[2], v)
proc vec4*[T](v2:Vec2[T], v:T, u:T):Vec4[T]= vec4( v2[0], v2[1], v, u)
proc vec4*[T](v2:Vec2[T], v22:Vec2[T]):Vec4[T]= vec4( v2[0], v2[1], v22[0], v22[1])
## TODO Add vec1 to contrustors

## GETTERS
proc x*[T](v:Vec1[T]):T = v[0]
proc x*[T](v:Vec2[T]):T = v[0]
proc x*[T](v:Vec3[T]):T = v[0]
proc x*[T](v:Vec4[T]):T = v[0]

proc y*[T](v:Vec2[T]):T = v[1]
proc y*[T](v:Vec3[T]):T = v[1]
proc y*[T](v:Vec4[T]):T = v[1]

proc z*[T](v:Vec3[T]):T = v[2]
proc z*[T](v:Vec4[T]):T = v[2]

proc w*[T](v:Vec4[T]):T = v[3]

## SETTERS
proc `x=`*[T](v:var Vec1[T],c:T)= v[0]=c
proc `x=`*[T](v:var Vec2[T],c:T)= v[0]=c
proc `x=`*[T](v:var Vec3[T],c:T)= v[0]=c
proc `x=`*[T](v:var Vec4[T],c:T)= v[0]=c

proc `y=`*[T](v:var Vec2[T],c:T)= v[1]=c
proc `y=`*[T](v:var Vec3[T],c:T)= v[1]=c
proc `y=`*[T](v:var Vec4[T],c:T)= v[1]=c

proc `z=`*[T](v:var Vec3[T],c:T)= v[2]=c
proc `z=`*[T](v:var Vec4[T],c:T)= v[2]=c

proc `w=`*[T](v:var Vec4[T],c:T)= v[3]=c


if isMainModule:
    echo vec1(10)
    echo vec2(10,15)
    echo vec2(10)
    echo vec3(10,15,20)
    echo vec3(0)
    echo vec4(10,15,20,35)
    echo vec4(50)
    var v1 = vec1()
       

    echo ("B:",v1)
    v1[0] = 10.0
    echo ("A:",  v1[0] )
    v1.x = 5.0;
    echo repr(addr v1)
    echo(v1)


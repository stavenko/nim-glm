import strutils, macros
import math
static:
    const MAX_VEC_SIZE:int = 4
type
    Vec1[T]= distinct array[1,T]
    Vec2[T]= distinct array[2,T]
    Vec3[T]= distinct array[3,T]
    Vec4[T]= distinct array[4,T]

proc `$`*[T](a:openArray[T]):string=
    var 
        strs:seq[string] = @[]
    for i in a:
        strs.add( $i);
    return [ "[" ,strs.join(", "),"]" ].join
proc zipWith[I,F,T](a, b:array[I,F], op:proc(a,b:F):T):array[I,T]=
    for i in low(a) .. high(a):
        result[i] = op(a[i],b[i])
template toA( v:expr, i:int, T:typedesc):expr=
    array[i,T](v)

macro mkMathPerComponent():stmt=
    let ops = ['+', '-', '/', '*']
    result = newNimNode(nnkStmtList);
    for i in 1..MAX_VEC_SIZE:
        for op in ops:
            var str:string = "proc `$1`[T](v,u:$2[T]):$2[T]= $2(zipWith( v.toA($3,T), u.toA($3,T), proc(a,b:T):T=a$1b))"%[$op, "Vec$#"%[$i], $i]
            result.add(parseStmt(str ))

macro mkToStr():stmt=
    result = newNimNode(nnkStmtList);
    for i in 1..MAX_VEC_SIZE:
        result.add(parseStmt("proc `$$`[T](v:Vec$1[T]):string=  $$ array[$1, T](v)" % [ $i ]))

macro arrSetters():stmt=
    result = newNimNode(nnkStmtList);
    for i in 1..MAX_VEC_SIZE:
        result.add(parseStmt("proc `[]=`[T](v:Vec$1[T], ix:int, c:T) = array[$1,T](v)[ix]=c" % [$i] ))

macro arrGetters():stmt=
    result = newNimNode(nnkStmtList);
    for i in 1..MAX_VEC_SIZE:
        result.add(parseStmt("proc `[]`[T](v:Vec$1[T], ix:int):T = array[$1,T](v)[ix]" % [$i] ))

macro componentGetterSetters():stmt=
    result = newNimNode(nnkStmtList);
    let templG = "proc $1[T](v:Vec$2[T]):T=array[$2,T](v)[$3]"
    let templS = "proc `$1=`[T](v:var Vec$2[T],c:T)=(array[$2,T](v))[$3]=c"
    let tr = ["x","y","z","w"]
    let col= ["r","g","b","a"]
    for i in low(tr)..high(tr):
        for s in 1..MAX_VEC_SIZE:
            if i<s:
                result.add(parseStmt( templS % [tr[i], $s, $i] ))
                result.add(parseStmt( templG % [tr[i], $s, $i] ))
                result.add(parseStmt( templS % [col[i], $s, $i] ))
                result.add(parseStmt( templG % [col[i], $s, $i] ))
proc pow(a,b:int):int= floor(pow(a.float, b.float)).int
proc fmod(a,b:int):int=floor(fmod(a.float, b.float)).int
iterator shifts(minArr,arrLen:int):seq[int]=
    for S in minArr..MAX_VEC_SIZE:
        var totCombInLoop = pow(arrLen,S) # Total combinations for alphabet length = arrLen and  selection size = S
        #echo("Collections:", totCombInLoop)
        for j in 0..totCombInLoop-1:
            var cs:seq[int] = @[]
            for i in 1..S:
                #echo("$#/$#/$#"%[$j, $(j.fmod(pow(arrLen,i))), $(pow(arrLen,i-1) )])
                cs.add( floor(j.fmod( pow(arrLen,i) )/pow(arrLen,i-1) ).int )
            yield cs

macro multiComponentGetterList():stmt=
    result = newNimNode(nnkStmtList);
    let comps=["x","y","z","w"]
    for i in 2..MAX_VEC_SIZE:
        for combination in shifts(2,i):
            var getter:seq[string] = @[]
            var arr:seq[string] = @[]
            for c in combination:
                getter.add(comps[c])
                arr.add("v.$1"% comps[c])
            var procStr = "proc $1[T](v: Vec$2[T]):Vec$3[T]=Vec$3[T]([$4])" % [getter.join(""), $i, $combination.len, arr.join(",")]
            result.add(parseStmt( procStr))

macro constructors():stmt=
    result = newNimNode(nnkStmtList);
    var 
        types:array[MAX_VEC_SIZE+1,string]
    proc glength(s:string):int=
        case s
        of "val", "1": result=1
        else: result = parseInt(s)
    types[0] = "val"
    for i in 1..MAX_VEC_SIZE:
        types[i] = $i
        
            
    



mkToStr()
mkMathPerComponent()
arrGetters()
arrSetters()
componentGetterSetters()
multiComponentGetterList()


if isMainModule:
    var 
        v=Vec4([2.0, 3.0, 0.0, 10.0] )
        v3=Vec3([3.0, 3.0, 3.0])
        vv = Vec1([1.0])
        u=Vec4([2.0, 3.0,5.0, 4.5] )
        a: array[5, float] = [0.0, 1.0, 2.0, 3.0, 4.0]
        b: array[5, float] = [0.0, 1.0, 2.0, 3.0, 4.0]
        c = v+u
        d = v*u
    v3.g = 4.0
    var vv4 = v3.yyyy
    echo ("V3", v3.yyyy)
    echo ("WHO:$$, $1"%["hi"])

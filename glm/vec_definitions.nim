import strutils
import sequtils
import macros
import algorithm
import math as m

import arrayUtils

export arrayUtils

#compile-time strongly inefficient math functions
proc floor(f:float):float=f.int.float
proc round(f:float):float=
    var flo = f.floor
    var d = f - flo
    if d >= 0.5: result = flo + 1.0
    else: result = flo

proc pow(a,b:float):float=
    result  = 1;
    for i in 1..b.int:
        result *= a
proc `fmod`(a,b:float):float=
    var diff = floor(a/b)*b
    return a-diff

macro defineVectorTypes*(upTo:int):stmt=
    var upToVec:int = intVal(upTo).int
    result = newNimNode(nnkStmtList);
    for i in 1 .. upToVec:
        var def = "type Vec$1*[T] = distinct array[$1, T]" % [$i]
        result.add(parseStmt(def))


template toA*( v:expr, i:int, T:typedesc):expr=
    array[i,T](v)

macro mkMathPerComponent*(upTo:int):stmt=
    var upToVec = intVal(upTo)
    let ops = ['+', '-', '/', '*']
    result = newNimNode(nnkStmtList);
    for i in 1..upToVec:
        for op in ops:
            var str:string = "proc `$1`*[T](v,u:$2[T]):$2[T]= $2(zipWith( v.toA($3,T), u.toA($3,T), proc(a,b:T):T=a$1b))"%[$op, "Vec$#"%[$i], $i]
            result.add(parseStmt(str ))

macro mkToStr*(upTo:int):stmt=
    var upToVec = intVal(upTo)
    result = newNimNode(nnkStmtList);
    for i in 1..upToVec:
        result.add(parseStmt("proc `$$`*[T](v:Vec$1[T]):string=  $$ array[$1, T](v)" % [ $i ]))

macro arrSetters*(upTo:int):stmt=
    var upToVec = intVal(upTo)
    result = newNimNode(nnkStmtList);
    for i in 1..upToVec:
        result.add(parseStmt("proc `[]=`*[T](v:var Vec$1[T], ix:int, c:T) = array[$1,T](v)[ix]=c" % [$i] ))

macro arrGetters*(upTo:int):stmt=
    var upToVec = intVal(upTo)
    result = newNimNode(nnkStmtList);
    for i in 1..upToVec:
        result.add(parseStmt("proc `[]`*[T](v:Vec$1[T], ix:int):T = array[$1,T](v)[ix]" % [$i] ))

macro componentGetterSetters*(upTo:int):stmt=
    var upToVec = intVal(upTo)
    result = newNimNode(nnkStmtList);
    let templG = "proc $1*[T](v:Vec$2[T]):T=array[$2,T](v)[$3]"
    let templS = "proc `$1=`*[T](v:var Vec$2[T],c:T)=(array[$2,T](v))[$3]=c"
    let tr = ["x","y","z","w"]
    let col= ["r","g","b","a"]
    for i in low(tr)..high(tr):
        for s in 1..upToVec:
            if i<s:
                result.add(parseStmt( templS % [tr[i], $s, $i] ))
                result.add(parseStmt( templG % [tr[i], $s, $i] ))
                result.add(parseStmt( templS % [col[i], $s, $i] ))
                result.add(parseStmt( templG % [col[i], $s, $i] ))
    
proc pow(a,b:int):int= floor(pow(a.float, b.float)).int
proc fmod(a,b:int):int=floor(fmod(a.float, b.float)).int
iterator shifts(minArr,arrLen,upToVec:int):seq[int]=
    for S in minArr..upToVec:
        var totCombInLoop = pow(arrLen,S) # Total combinations for alphabet length = arrLen and  selection size = S
        for j in 0..totCombInLoop-1:
            var cs:seq[int] = @[]
            for i in 1..S:
                cs.add( floor(j.fmod( pow(arrLen,i) )/pow(arrLen,i-1) ).int )
            yield cs

macro multiComponentGetterList*(upTo:int):stmt=
    var upToVec = intVal(upTo)
    result = newNimNode(nnkStmtList);
    let comps=["x","y","z","w"]
    for i in 2..upToVec.int:
        for combination in shifts(2,i, upToVec.int):
            var getter:seq[string] = @[]
            var arr:seq[string] = @[]
            for c in combination:
                getter.add(comps[c])
                arr.add("v.$1"% comps[c])
            var procStr = "proc $1*[T](v: Vec$2[T]):Vec$3[T]=Vec$3[T]([$4])" % [getter.join(""), $i, $combination.len, arr.join(",")]
            result.add(parseStmt( procStr))

        
proc getComponentIx(shiftn,length,componentIx:int):int=
    floor(shiftn.fmod( pow(length,componentIx) )/pow(length,componentIx-1) ).int
    
proc shifts(data:seq[char]):seq[seq[char]]=
    
    var length = data.len;
    var shifts = pow(length, length)
    result = @[]
    
    for i in 0..shifts-1:
        var cs:seq[char] = @[]
        for j in 1..length:
            cs.add( data[getComponentIx(i, length, j)] )
        result.add(cs)

proc flatten[T](s:seq[seq[T]]):seq[T]=
    result = @[]
    for i in s:
        result = result & i
proc getCombinationsForLength(vecLength,upToVec:int):seq[seq[char]]=
    var vx:seq[char] = @['V']
    for i in 1..upToVec:
        var c:char = ($i)[0]
        vx.add(c)
    proc glength(s:char):int=
        case s
        of 'V', '1': result=1
        else: result = parseInt($s)
    var available = vx.filter(proc(x:char):bool=return glength(x) <= vecLength)
    result = @[]
    var count = 0;
    for sym in available.reversed():
        var res:seq[char] = @[ sym ]
        count += glength(sym)
        if count < vecLength:
            var newSymbols = getCombinationsForLength(vecLength - count, upToVec)
            for s in newSymbols:
                var nres = res.concat(s)
                result.add(nres)
            count =0
        else:
            count = 0
            result.add(res);

proc componentGetters(c:char, sm:string):seq[string]=
    case c
    of 'V': result = @[sm]
    else:
        result = @[]
        var size = parseInt($c)
        for i in 0..size-1:
            result.add("$1[$2]" % [ sm, $i])


proc parameterConstructor(c:char, sm:string):seq[string]=
    case c
    of 'V': result = @["$1:T" % [sm] ]
    else:
        result = @[]
        var size = parseInt($c)
        result.add("$1:Vec$2[T]" % [sm, $size ])


macro createConstructors*(upTo:int):stmt =
    var upToVec = intVal(upTo)
    result = newNimNode(nnkStmtList);
    let paramnames = "abcdefghijklomnpo"
    for vl in 1..upToVec.int:
        var procStr = "proc vec$1*[$4]($2):Vec$1[$4]=Vec$1([$3])"
        var procStrU = "proc vec$1*($2):Vec$1[$4]=Vec$1([$3])"
        # create empty constructor
        var resultProc = procStrU % [$vl, "", repeat(@["0.0"],vl).join(", "), "float" ]
        result.add(parseStmt(resultProc))
        # create one parameter constructor
        if(vl > 1):
            resultProc = procStr % [$vl, "a:T", repeat(@["a"],vl).join(", "), "T" ]
            result.add(parseStmt(resultProc))
        for combination in getCombinationsForLength(vl, upToVec.int):
            var arrayConstructor:seq[string] = @[]
            var parameterConstructor:seq[string] = @[]
            for ix in combination.low..combination.high:
                for cg in componentGetters(combination[ix], $(paramnames[ix])):
                    arrayConstructor.add(cg)
                for pc in parameterConstructor(combination[ix], $(paramnames[ix])):
                    parameterConstructor.add(pc)
            var resultProc = procStr % [
                $vl,
                parameterConstructor.join(", "),
                arrayConstructor.join(", "),
                "T"
            ]
            result.add(parseStmt( resultProc) )


proc `*`*[T](a:var array[3,T], s:T)=
    for i in 0..2: a[i] *= s
    


macro createScalarOperations*(upTo:int):stmt=
    let upToVec = intVal(upTo).int
    let ops = ["+", "-", "/", "*"]
    result = newNimNode(nnkStmtList);
    for vs in 1 .. upToVec:
        for op in ops:
            var procs = "proc `$1`*[T](a:Vec$2[T], s:T):Vec$2[T]=Vec$2(map(array[$2,T](a), proc(a:T):T=a $1 s))" % [ op, $vs ]
            var inlProcs = """proc `$1=`*[T](a:var Vec$2[T], s:T)=
            for i in 0..$2-1:
                a[i] = a[i] $1 s """ % [ op, $vs ]
            result.add(parseStmt(procs))
            result.add(parseStmt(inlProcs))


macro createDotProduct*(upTo:int):stmt=
    result = newNimNode(nnkStmtList);
    for i  in 1..upTo.intVal.int:
        let dotProducts = "proc dot*[T](a,b:Vec$1[T]):T=sum(toA(a*b,$1,T))" % [ $i ]
        result.add(parseStmt(dotProducts))



macro createLengths*(upTo:int):stmt=
    result = newNimNode(nnkStmtList);
    for i  in 1..upTo.intVal.int:
        let l2s = "proc len2*[T](a:Vec$1[T]):T=sum(toA(a*a,$1,T))" % [ $i ]
        let ls = "proc len*[T](a:Vec$1[T]):T=sqrt(a.len2)" % [ $i ]
        result.add(parseStmt(l2s))
        result.add(parseStmt(ls))




import sequtils
import strutils
proc `$`*[T](a:openArray[T]):string=
    var 
        strs:seq[string] = @[]
    for i in a:
        strs.add( $i);
    return [ "[" ,strs.join(", "),"]" ].join



proc matProduct*[N,M,P,T](a:array[N,array[M,T]],
                         b:array[M,array[P,T]]):array[N,array[P,T]]=
    var 
        n = a.len-1
        m = b.len-1
        p = b[0].len-1
    for c in 0..n:
        for r in 0..p:
            result[c][r] = 0
            for i in 0..m:
                result[c][r] += a[c][i] * b[i][r]


proc matVecProduct*[N,M,T](mat:array[N,array[M,T]], v:array[N,T]):array[M,T]=
    var
        n = mat.len
        m = mat[0].len
    for i in 0..m-1:
        result[i] = 0
        for c in 0..n-1:
            result[i] += mat[c][i] * v[c]
proc matVecProduct*[N,M,T](v:array[M,T], mat:array[N,array[M,T]]):array[N,T]=
    var 
        n = mat.len
        m = mat[0].len
    for i in 0..n-1:
        result[i] = 0
        for r in 0..m-1:
            result[i] += mat[i][r] * v[r]


proc zipWith*[I,F,T](a, b:array[I,F], op:proc(a,b:F):T):array[I,T]=
    for i in low(a) .. high(a):
        result[i] = op(a[i],b[i])

proc `$`*[N,M,T](arr:array[N,array[M,T]]):string=
    var 
        cols = arr.len
        rows = arr[0].len
        matStr :seq[string] = @[]
    for row in 0..rows-1:
        var rowStr:seq[string] = @[]
        for col in 0..cols-1:
            rowStr.add($arr[col][row])
        matStr.add(rowStr.join "  ")
    result = "\n" & matStr.join("\n")

proc map*[N,T](a:array[N,T], f:proc(a:T):T):array[N,T]=
    for i in a.low .. a.high:
        result[i] = f(a[i])


proc foldl[N,T,S](a:array[N,T], acc:S, f:proc(a:S,b:T):S):S=
    result = acc
    for i in a.low .. a.high:
        result = f(result, a[i])
        


    

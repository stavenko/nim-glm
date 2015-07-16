import sequtils
import strutils
proc `$`*[T](a:openArray[T]):string=
    var 
        strs:seq[string] = @[]
    for i in a:
        strs.add( $i);
    return [ "[" ,strs.join(", "),"]" ].join



proc matProduct[N,M,P,T](a:array[N,array[M,T]],
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


proc zipWith*[I,F,T](a, b:array[I,F], op:proc(a,b:F):T):array[I,T]=
    for i in low(a) .. high(a):
        result[i] = op(a[i],b[i])

proc `$`*[N,M,T](arr:array[N,array[M,T]]):string=
    var matStr :seq[string] = @[]
    for row in arr.low .. arr.high:
        var rowStr:seq[string] = @[]
        var innerArr = arr[row]
        for col in innerArr.low..innerArr.high:
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
        

if isMainModule:
    var m1 = [[1,0],[1,1]]
    var m2 = [[1,0],[0,1]]

    echo m1, m1.matProduct( m2)
#proc sum*[T](a:openarray[T]):T=foldlS(a, 0, proc(a,b:T):T=a+b)

    

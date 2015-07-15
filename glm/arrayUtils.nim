import sequtils
proc `$`*[T](a:openArray[T]):string=
    var 
        strs:seq[string] = @[]
    for i in a:
        strs.add( $i);
    return [ "[" ,strs.join(", "),"]" ].join

proc zipWith*[I,F,T](a, b:array[I,F], op:proc(a,b:F):T):array[I,T]=
    for i in low(a) .. high(a):
        result[i] = op(a[i],b[i])

proc `$`*[N,T](arr:array[N,array[N,T]]):string=
    var matStr :seq[string] = @[]
    for row in 0..3:
        var rowStr:seq[string] = @[]
        for col in 0..3:
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
        

#proc sum*[T](a:openarray[T]):T=foldlS(a, 0, proc(a,b:T):T=a+b)

    

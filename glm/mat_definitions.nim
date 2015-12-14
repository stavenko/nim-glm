import macros, strutils

template macroInit(m,M:expr){.immediate.}=
    result = newNimNode(nnkStmtList);
    var 
        m = minSize.intVal.int
        M = maxSize.intVal.int

macro defineMatrixTypes*(minSize,maxSize: int):stmt=
    macroInit(m, M)
    for col in  m .. M:
        for row in m .. M:
            var def = "type Mat$1x$2*[T] = distinct array[$1, Vec$2[T]]" % [$col, $row]
            result.add(parseStmt(def))

macro matrixEchos*(minSize, maxSize:int):stmt=
    macroInit(m, M)
    for col in m..M:
        for row in m .. M:
            var def = "proc `$$`*[T](m:Mat$1x$2[T]):string = $$ array[$1, array[$2,T]](m)" % [$col, $row]
            result.add(parseStmt(def))

macro addrGetter*(minSize, maxSize:int):stmt=
    macroInit(m, M)
    let procT = "proc addr*[T](m:var Mat$1x$2[T]):ptr T= array[$1, array[$2,T]](m)[0][0].addr"
    for col in m..M:
        for row in m .. M:
            var def = procT % [ $col, $row]
            result.add(parseStmt(def))

macro columnGetters*(minSize, maxSize:int):stmt=
    macroInit(m, M)
    let procT = "proc `[]`*[T](m:Mat$1x$2[T], ix:int):Vec$2[T]= array[$1, Vec$2[T]](m)[ix]"
    let procTvar = "proc `[]`*[T](m:var Mat$1x$2[T], ix:int):var Vec$2[T]= array[$1, Vec$2[T]](m)[ix]"
    for col in m..M:
        for row in m .. M:
            var def1 = procT % [ $col, $row]
            var def2 = procTvar % [ $col, $row]
            result.add(parseStmt(def1))
            result.add(parseStmt(def2))
            
macro columnSetters*(minSize, maxSize:int):stmt=
    macroInit(m, M)
    let procT = "proc `[]=`*[T](m:var Mat$1x$2[T], ix:int, c:Vec$2[T])= array[$1, Vec$2[T]](m)[ix] = c"
    for col in m..M:
        for row in m .. M:
            var def = procT % [ $col, $row]
            result.add(parseStmt(def))

macro matrixScalarOperations*(minSize, maxSize:int):stmt=
    macroInit(m, M)
    let procT = "proc `$3`*[T](m:Mat$1x$2[T], s:T):Mat$1x$2[T]=" &
                 "Mat$1x$2(map(array[$1,Vec$2[T]](m),proc(v:Vec$2[T]):Vec$2[T]= v $3 s))" 
    for op in ["+", "-", "*", "/"]:
        for col in m..M:
            for row in m .. M:
                var def = procT % [ $col, $row, op ]
                result.add(parseStmt(def))
macro matrixUnaryScalarOperations*(minSize, maxSize:int):stmt=
    macroInit(m,M)
    let opT = "    m[$1][$2]= m[$1][$2] $3  s"
    let T = "proc `$3=`*[T](m:var Mat$1x$2[T], s:T)=\n    var a = array[$1,array[$2,T]](m)\n"
    
    for op in ["+", "-", "*", "/"]:
        for col in m..M:
            for row in m..M:
                var mm:seq[string] = @[]
                var def = T % [$col, $row, op]
                for i in 0..col-1:
                    for j in 0..row-1:
                        mm.add(opT % [$i, $j, op])
                def &= mm.join("\n")
                result.add(parseStmt(def))

                

# we need matrix constructors
macro matrixConstructors*(minSize, maxSize:int):stmt=
    macroInit(m, M)
    let vars = ["a","b","c","d"]
    let procTemplate = "proc mat$1x$2*[T]($3):Mat$1x$2[T]=" &
                       "Mat$1x$2([$4])"
    let procTemplateS = "proc mat$1*[T]($3):Mat$1x$2[T]=" &
                        "Mat$1x$2([$4])"
    for col in m..M:
        for row in m..M:
            var fvecs:seq[string] = @[]
            for i in 0..col-m+1:
                fvecs.add(vars[i])
            var finput = "$#:Vec$#[T]" % [fvecs.join(","), $row]
            if row == col:
                let constr = procTemplateS % [ $col,
                                               $row,
                                               finput,
                                               fvecs.join(", ") ]
                result.add(parseStmt(constr))

            let constr = procTemplate % [ $col,
                                          $row,
                                          finput,
                                          fvecs.join(", ") ]
            result.add(parseStmt(constr))
#macro copyConstructors*(minSize, maxSize:int):stmt=
    #macroInit(m, M)
    
    #for col in m..M:
        #for row in m..M:
macro diagonalConstructors*(minSize,maxSize:int):stmt=
    macroInit(m, M)
    let T = "proc mat$1x$2*[T](s:T):Mat$1x$2[T]=mat$1x$2($3)"    
    let Tt = "proc mat$1*[T](s:T):Mat$1x$2[T]=mat$1x$2($3)"    
    var vT = "vec$1($2)"
    for col in m..M:
        for row in m..M:
            var vv:seq[string] = @[]
            for c in 0..col-1:
                var v:seq[string] = @[]
                for r in 0..row-1:
                    v.add(if c==r: "s" else: "0.T")
                vv.add(vT % [ $row, v.join(", ")])
            if(row == col):
                result.add(parseStmt(Tt %[$col, $row, vv.join(", ")]))
            var def = T %[$col, $row, vv.join(", ")]
            result.add(parseStmt(def))


macro emptyConstructors*(minSize, maxSize:int):stmt=
    macroInit(m,M)
    let vecTemplate = "vec$1($2)" 
    let fullTemplate= "proc mat$1x$2*():Mat$1x$2[float]=mat$1x$2($3)"
    let partialTemplate= "proc mat$1*():Mat$1x$2[float]=mat$1($3)"
    for col in m..M:
        for row in m..M:
            var vecs :seq[string] = @[]
            for i in 0..col-m+1:
                var vecComponents:seq[string] = @[]
                for j in 0..row-m+1:
                    var c = if i==j: "1.0" else: "0.0"
                    vecComponents.add(c)
                vecs.add(vecTemplate % [$row, vecComponents.join(", ")])
            let matProc =  fullTemplate % [$col, $row, vecs.join(", ")]
            if col == row:
                let f = partialTemplate % [$col, $row, vecs.join(", ")]
                result.add(parseStmt(f))
            result.add(parseStmt(matProc))

macro fromArray*(minSize, maxSize:int):stmt=
    macroInit(m, M)
    var T = "proc mat$1x$2*[T](a:array[$1,array[$2,T]]):Mat$1x$2[T]=Mat$1x$2([$3])"
    var Tv = "Vec$1(a[$2])"
    for col in m..M:
        for row in m..M:
            var vectors:seq[string] = @[]
            for c in 0..col-1:
                vectors.add(Tv % [ $row, $c])

            var def = T% [ $col, $row, vectors.join(", ") ]
            result.add(parseStmt( def ))




macro matrixMultiplication*(minSize, maxSize:int):stmt=
    macroInit(m,M)
    let Template = "proc `*`*[T](a:Mat$1x$2[T], b:Mat$2x$3[T]):Mat$1x$3[T]=" &
                        "matProduct(array[$1, array[$2,T]](a), array[$2,array[$3,T]](b)).mat$1x$3"
    var tuples:seq[tuple[c:int,r:int]] = @[]
    for col in m..M:
        for row in m..M:
            tuples.add( (col, row) )
    for l in tuples:
        for r in tuples:
            if l.r == r.c:
                var def = Template % [$l.c, $l.r, $r.r]
                result.add(parseStmt( def ));
            
macro matrixVectorMultiplication*(minSize, maxSize:int):stmt=
    macroInit(m,M)
    let Tv = "proc `*`*[T](m:Mat$1x$2[T], v:Vec$1[T]):Vec$2[T]=Vec$2(matVecProduct( array[$1,array[$2,T]](m), array[$1,T](v)))"
    let vT = "proc `*`*[T](v:Vec$2[T], m:Mat$1x$2[T] ):Vec$1[T]=Vec$1(matVecProduct(array[$2,T](v), array[$1,array[$2,T]](m)))"
    for col in m..M:
        for row in m..M:
            let def1 = Tv % [ $col, $row ]
            let def2 = vT % [ $col, $row ]
            result.add(parseStmt( def1 ));
            result.add(parseStmt( def2 ));



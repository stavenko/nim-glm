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
            var def = "type Mat$1x$2[T] = distinct array[$1, Vec$2[T]]" % [$col, $row]
            echo def
            result.add(parseStmt(def))

macro matrixEchos*(minSize, maxSize:int):stmt=
    macroInit(m, M)
    for col in m..M:
        for row in m .. M:
            var def = "proc `$$`*[T](m:Mat$1x$2[T]):string = $$ array[$1, array[$2,T]](m)" % [$col, $row]
            echo def
            result.add(parseStmt(def))

macro addrGetter*(minSize, maxSize:int):stmt=
    macroInit(m, M)
    let procT = "proc addr*[T](m:var Mat$1x$2[T]):ptr T= array[$1, array[$2,T]](m)[0][0].addr"
    for col in m..M:
        for row in m .. M:
            var def = procT % [ $col, $row]
            echo def
            result.add(parseStmt(def))
macro columnGetters*(minSize, maxSize:int):stmt=
    macroInit(m, M)
    let procT = "proc `[]`*[T](m:var Mat$1x$2[T], ix:int):var Vec$2[T]= array[$1, Vec$2[T]](m)[ix]"
    for col in m..M:
        for row in m .. M:
            var def = procT % [ $col, $row]
            echo def
            result.add(parseStmt(def))
            
macro columnSetters*(minSize, maxSize:int):stmt=
    macroInit(m, M)
    let procT = "proc `[]=`*[T](m:var Mat$1x$2[T], ix:int, c:Vec$2[T])= array[$1, Vec$2[T]](m)[ix] = c"
    for col in m..M:
        for row in m .. M:
            var def = procT % [ $col, $row]
            echo def
            result.add(parseStmt(def))
macro matrixScalarOperations*(minSize, maxSize:int):stmt=
    macroInit(m, M)
    let procT = "proc `$3`*[T](m:var Mat$1x$2[T], s:T):Mat$1x$2[T]=" & 
                 "Mat$1x$2(map(array[$1,Vec$2[T]](m),proc(v:Vec$2[T]):Vec$2[T]= v $3 s))" 
    for op in ["+", "-", "*", "/"]:
        for col in m..M:
            for row in m .. M:
                var def = procT % [ $col, $row, op ]
                echo def
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
                echo constr
                result.add(parseStmt(constr))

            let constr = procTemplate % [ $col,
                                          $row,
                                          finput,
                                          fvecs.join(", ") ]
            echo constr
            result.add(parseStmt(constr))
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
                echo f
                result.add(parseStmt(f))
            result.add(parseStmt(matProc));

macro matrixMultiplication*(minSize, maxSize:int):stmt=
    macroInit(m,M)
    let Template = "proc `*`*[T](a:Mat$1x$2[T], b:Mat$2x$3[T]):Mat$1x$3[T]=" &
                        "matProduct(array[$1, array[$2,T]](a), array[$2,array[$3,T]](b) )"
    for col1 in m..M:
        for col2 in m..M:
            for row1 in m..M:
                for row2 in m..M:
                    if(row1 == col2):
                        var def = Template % [$col1, $row1, $col2] 
                        echo def
            

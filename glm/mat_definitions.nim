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
            result.add(parseStmt(def))

macro matrixEchos*(minSize, maxSize:int):stmt=
    macroInit(m, M)
    for col in m..M:
        for row in m .. M:
            var def = "proc `$$`*[T](m:Mat$1x$2):string = $$ array[$1, array[$2,T]](m)" % [$col, $row]
            result.add(parseStmt(def))


# we need matrix constructors
macro matrixConstructors*(minSize, maxSize:int):stmt=
    macroInit(m, M)
    let vars = ["a","b","c","d"]
    let procTemplate = "proc mat$1x$2*[T]($3):Mat$1x$2[T]=" &
                       "Mat$1x$2(array[$1,[$4]])"
    let procTemplateS = "proc mat$1*[T]($3):Mat$1x$2[T]=" &
                        "Mat$1x$2(array[$1,[$4]])"
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
macro emptyConstructors*(minSize, maxSize:int):stmt=
    macroInit(m,M)
    for col in m..M:
        for row in m..M:
            echo "mat$1x$2*():Mat$1x$2[float] = (" %[ $col, $row]


    

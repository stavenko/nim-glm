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
            var def = "proc `$`*[T](m:Mat$1x$2):string = $array($1, array[$2,T]](m)" % [$col, $row]



import macros
import strutils
import arrayUtils

macro normalizeMacros*(vectorSize:int):stmt=
    var upTo = vectorSize.intVal.int
    result = newNimNode(nnkStmtList);
    var T = "proc normalize*[T](v:Vec$1[T]):Vec$1[T]= v / v.length"
    for i in 1.. upTo:
        result.add(parseStmt(T % [$i]));




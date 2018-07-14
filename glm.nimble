#[ Package ]#

version       = "1.1.1"
author        = "Arne Döring"
description   = "Nim port of the OpenGL Mathematics (GLM) library"
license       = "MIT"

skipDirs = @["tests", "examples"]

requires("nim >= 0.17.2")

task test, "Runs the test suite":
  exec "nim c -r tests/testrunner.nim"

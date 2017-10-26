import glm, sdl2/sdl, opengl
import unittest

proc compare*[N,M,T](a,b: Mat[N, M, T]): bool =
  for i in 0 ..< N:
    for j in 0 ..< M:
      if abs(a[i][j]-b[i][j]) > 1e-5:
        return false;
  return true

proc getModelViewMatrix(): Mat4f =
  glGetFloatv(GL_MODELVIEW_MATRIX, result[0,0].addr)

suite "matrix transform":
  ## this test ensures that mat_transform functions behave identical to
  ## OpenGL functions, therefore it needs an OpenGL context

  setup:
    var window: Window
    var context: GLContext

    discard sdl.init(INIT_EVERYTHING)

    doAssert 0 == glSetAttribute(GL_CONTEXT_MAJOR_VERSION, 1)
    doAssert 0 == glSetAttribute(GL_CONTEXT_MINOR_VERSION, 0)

    let posx: cint = WINDOWPOS_UNDEFINED
    let posy: cint = WINDOWPOS_UNDEFINED

    # we need an opengl context to compare the results, but we do not
    # actually need to see the window, so it's hidden.
    let flags: uint32 = sdl.WINDOW_HIDDEN or sdl.WINDOW_OPENGL
    window = createWindow("window title", posx, posy, 640, 480, flags )

    if window.isNil:
      quit($sdl.getError())

    context = window.glCreateContext()
    if context.isNil:
      quit($sdl.getError())

    #Initialize OpenGL
    loadExtensions()

    doAssert 0 == glMakeCurrent(window, context)

    glMatrixMode(GL_MODELVIEW)

    var alpha: float32 = 123
    var n: Vec3f = vec3f(-7,8,-9).normalize
    var mat1, mat2: Mat4f

  test "translate":
    mat1 = mat4f(1)
      .translate(1,2,3)

    glLoadIdentity()
    glTranslatef(1,2,3)

    mat2 = getModelViewMatrix()
    check compare(mat1, mat2)

  test "scale":
    mat1 = mat4f(1)
      .scale(4,5,6)

    glLoadIdentity()
    glScalef(4,5,6)

    mat2 = getModelViewMatrix()

    check compare(mat1, mat2)

  test "rotate":
    mat1 = mat4f(1)
      .rotate(alpha, n.x, n.y, n.z)

    glMatrixMode(GL_MODELVIEW)

    glLoadIdentity()
    glRotatef(alpha * 180 / Pi, n.x, n.y, n.z)

    glGetFloatv(GL_MODELVIEW_MATRIX, mat2[0,0].addr)

    check compare(mat1, mat2)

  test "combined A":
    mat1 = mat4f(1)
      .translate(1,2,3)
      .scale(4,5,6)
      .rotate(alpha, n.x, n.y, n.z)

    glMatrixMode(GL_MODELVIEW)

    glLoadIdentity()
    glTranslatef(1,2,3)
    glScalef(4,5,6)
    glRotatef(alpha * 180 / Pi, n.x, n.y, n.z)

    glGetFloatv(GL_MODELVIEW_MATRIX, mat2[0,0].addr)

    check compare(mat1, mat2)

  test "combiend B":
    ## tests a permutation of the matrix operations
    mat1 = mat4f(1)
      .rotate(alpha, n.x, n.y, n.z)
      .scale(4,5,6)
      .translate(1,2,3)

    glMatrixMode(GL_MODELVIEW)

    glLoadIdentity()
    glRotatef(alpha * 180 / Pi, n.x, n.y, n.z)
    glScalef(4,5,6)
    glTranslatef(1,2,3)

    glGetFloatv(GL_MODELVIEW_MATRIX, mat2[0,0].addr)

    doAssert compare(mat1, mat2)

  teardown:
    sdl.gl_DeleteContext(context)
    sdl.destroyWindow(window)
    sdl.quit()

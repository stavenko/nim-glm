import ../glm, sdl2/sdl, opengl

## this test ensures that mat_transform functions behave identical to
## OpenGL functions, therefore it needs an OpenGL context

var window: Window
var context: GLContext

block setup:
  discard sdl.init(INIT_EVERYTHING)

  doAssert 0 == glSetAttribute(GL_CONTEXT_MAJOR_VERSION, 1)
  doAssert 0 == glSetAttribute(GL_CONTEXT_MINOR_VERSION, 0)

  let posx = WINDOWPOS_UNDEFINED.cint
  let posy = WINDOWPOS_UNDEFINED.cint

  window = createWindow("window title", posx, posy, 640, 480, WINDOW_OPENGL)

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

proc getModelViewMatrix(): Mat4f =
  glGetFloatv(GL_MODELVIEW_MATRIX, result[0,0].addr)

proc matrixCompare(arg1, arg2: Mat4f): float32 =
  for i in 0..3:
    for j in 0..3:
      result += abs(arg1[i,j] - arg2[i,j])

block testTranslate:
  mat1 = mat4f(1)
    .translate(1,2,3)

  glLoadIdentity()
  glTranslatef(1,2,3)

  mat2 = getModelViewMatrix()
  doAssert matrixCompare(mat1, mat2) < 1e5

block testScale:
  mat1 = mat4f(1)
    .scale(4,5,6)

  glLoadIdentity()
  glScalef(4,5,6)

  mat2 = getModelViewMatrix()

  doAssert matrixCompare(mat1, mat2) < 1e5

block testRotate:
  mat1 = mat4f(1)
    .rotate(alpha, n.x, n.y, n.z)

  glMatrixMode(GL_MODELVIEW)

  glLoadIdentity()
  glRotatef(alpha * 180 / Pi, n.x, n.y, n.z)

  glGetFloatv(GL_MODELVIEW_MATRIX, mat2[0,0].addr)

  doAssert matrixCompare(mat1, mat2) < 1e5

block testAll:
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

  doAssert matrixCompare(mat1, mat2) < 1e5

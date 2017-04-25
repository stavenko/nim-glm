#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/io.hpp>
#include <iostream>

using namespace std;
using namespace glm;

int main() {
  float a = 1.0f / sqrt(2.0f);
  float b = 1.0f / sqrt(3.0f);

  glm::quat quaternions[] = {
    quat(0,1,0,0), quat(0,0,1,0), quat(0,0,0,1), quat(1,0,0,0),
    quat(0,a,a,0), quat(0,a,0,a), quat(a,a,0,0), quat(0,0,a,a), quat(a,0,a,0), quat(a,0,0,a),
    quat(b,0,b,b), quat(b,b,0,b), quat(b,b,b,0), quat(0,b,b,b)
  };

  for ( quat q : quaternions ) {
    mat3 mat = mat3_cast(q);
    cout << mat << endl;
    quat q2 = quat_cast(mat3_cast(q));
    q2[0] -= q[0];
    q2[1] -= q[1];
    q2[2] -= q[2];
    q2[3] -= q[3];

    cout << q2 << endl;

  }
}

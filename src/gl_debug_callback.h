#pragma once
#include <Open3D/Open3D.h>
#include <stdio.h>

void APIENTRY GLDebugMessageCallback(GLenum source, GLenum type, GLuint id,
                            GLenum severity, GLsizei length,
                            const GLchar *msg, const void *data);
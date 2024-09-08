#pragma once
#include <open3d/Open3D.h>
#include <stdio.h>

void APIENTRY GLDebugMessageCallback(GLenum source, GLenum type, GLuint id,
                            GLenum severity, GLsizei length,
                            const GLchar *msg, const void *data);
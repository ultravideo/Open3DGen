#include "shader_stuff.h"

#ifdef OPEN3DGEN_HEADLESS 
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>
EGLContext ctx;
EGLDisplay eglDpy;
  static const EGLint configAttribs[] = {
          EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
          EGL_BLUE_SIZE, 8,
          EGL_GREEN_SIZE, 8,
          EGL_RED_SIZE, 8,
          EGL_DEPTH_SIZE, 8,
          EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
          EGL_NONE
  };    

  static const int pbufferWidth = 10;
  static const int pbufferHeight = 10;

  static const EGLint pbufferAttribs[] = {
        EGL_WIDTH, pbufferWidth,
        EGL_HEIGHT, pbufferHeight,
        EGL_NONE,
  };
#endif



namespace stitcher3d
{
namespace shader
{

void create_compute_shader()
{

}

bool get_gl_error(GLuint shader_i)
{
    GLenum invalid = glGetError();
    if (invalid == GL_NO_ERROR)
        return true;
    std::cout << "0x" << std::hex << invalid << " enum value\n";

    // here for hex values values
    // https://www.khronos.org/opengl/wiki/OpenGL_Error

    glDeleteShader(shader_i);

    return false;
}

std::shared_ptr<std::vector<pixel_data>> dispatch_compute_shader(shader_info& s_info)
{
    #ifdef OPEN3DGEN_HEADLESS
    PFNGLUSEPROGRAMPROC glUseProgram =  (PFNGLUSEPROGRAMPROC)eglGetProcAddress("glUseProgram");
    PFNGLGENBUFFERSPROC  glGenBuffers = (PFNGLGENBUFFERSPROC ) eglGetProcAddress("glGenBuffers");
    PFNGLBINDBUFFERPROC glBindBuffer = (PFNGLBINDBUFFERPROC) eglGetProcAddress("glBindBuffer");
    PFNGLBINDBUFFERBASEPROC glBindBufferBase = (PFNGLBINDBUFFERBASEPROC) eglGetProcAddress("glBindBufferBase");
    PFNGLBUFFERDATAPROC glBufferData = (PFNGLBUFFERDATAPROC) eglGetProcAddress("glBufferData");
    PFNGLTEXSTORAGE2DPROC glTexStorage2D = (PFNGLTEXSTORAGE2DPROC) eglGetProcAddress("glTexStorage2D");
    PFNGLUNIFORM1IPROC glUniform1i = (PFNGLUNIFORM1IPROC) eglGetProcAddress("glUniform1i");
    PFNGLUNIFORM1FPROC glUniform1f = (PFNGLUNIFORM1FPROC) eglGetProcAddress("glUniform1f");
    PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC) eglGetProcAddress("glGetUniformLocation");
    PFNGLUNIFORMMATRIX4FVPROC glUniformMatrix4fv = (PFNGLUNIFORMMATRIX4FVPROC) eglGetProcAddress("glUniformMatrix4fv");
    PFNGLDISPATCHCOMPUTEPROC glDispatchCompute = (PFNGLDISPATCHCOMPUTEPROC)eglGetProcAddress("glDispatchCompute");
    PFNGLMEMORYBARRIERPROC glMemoryBarrier = (PFNGLMEMORYBARRIERPROC)eglGetProcAddress("glMemoryBarrier");
    PFNGLMAPBUFFERRANGEPROC glMapBufferRange = (PFNGLMAPBUFFERRANGEPROC)eglGetProcAddress("glMapBufferRange");
    PFNGLUNMAPBUFFERPROC glUnmapBuffer = (PFNGLUNMAPBUFFERPROC)eglGetProcAddress("glUnmapBuffer");
    PFNGLDELETEBUFFERSPROC glDeleteBuffers = (PFNGLDELETEBUFFERSPROC)eglGetProcAddress("glDeleteBuffers");
    #endif
    glUseProgram(s_info.shader_program);
        assert(get_gl_error(s_info.shader_program));

    const uint img_width = s_info.img_width;
    const uint img_height = s_info.img_height;

    // setup hit point output buffer
    pixel_data* pdata = new pixel_data[img_width*img_height];
    for (int i = 0; i < img_width*img_height; i++)
    {
        pdata[i] = pixel_data {
          -3.f, -3.f, -3.f,
          -3,
          -3, -3,
          0, 0
        };
    }

    GLuint hit_data_ssbo;
    GLuint tr_ssbo;
    GLuint d_tex;
    // set buffers
    {
        glGenBuffers(1, &hit_data_ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, hit_data_ssbo);
            assert(get_gl_error(s_info.shader_program));
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(pixel_data) * img_width*img_height, pdata, GL_DYNAMIC_COPY);
            assert(get_gl_error(s_info.shader_program));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, hit_data_ssbo);
            assert(get_gl_error(s_info.shader_program));
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        // setup triangle input buffer
        glGenBuffers(1, &tr_ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, tr_ssbo);
            assert(get_gl_error(s_info.shader_program));
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(vertex_data) * s_info.triangle_buffer_size, &s_info.triangle_buffer->at(0), GL_DYNAMIC_COPY);
            assert(get_gl_error(s_info.shader_program));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, tr_ssbo);
            assert(get_gl_error(s_info.shader_program));
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        // bind depth image to buffer
        glGenTextures(1, &d_tex);
        glBindTexture(GL_TEXTURE_2D, d_tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, img_width, img_height);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, img_width, img_height, GL_RED, GL_FLOAT, &s_info.depth_buffer->at(0));
            assert(get_gl_error(s_info.shader_program));
        

        glUniform1i(glGetUniformLocation(s_info.shader_program, "triangle_count"), s_info.triangle_buffer_size);
            assert(get_gl_error(s_info.shader_program));
        
        glUniform1f(glGetUniformLocation(s_info.shader_program, "depth_cloud_world_scale"), s_info.depth_cloud_world_scale);
            assert(get_gl_error(s_info.shader_program));
            
        glUniformMatrix4fv(glGetUniformLocation(s_info.shader_program, "camera_transform"), 1, GL_FALSE, &(s_info.camera_transform[0]));
            assert(get_gl_error(s_info.shader_program));
    }

    // 900 because local workgroup is of size [1024, 1, 1] and one thread per pixel
    // --> 1280*720 / 1536 = 900
    // for 1920x1080: 1920*1080 / 1536 = 1350
    // glDispatchCompute(600, 1, 1);
    // glDispatchCompute(WORK_GROUP_IMAGE_SIZE_LOOKUP.at(img_width*img_height), 1, 1);

    /**
     *  TODO:   
     *      on GTX1080 x, y, z, inv : 2147483647, 65535, 65535, 1536.
     *      maybe use only x -axis as linear? Limitations on slower hardware?
     */
    glDispatchCompute(s_info.workgroup_size, 1, 1);
        assert(get_gl_error(s_info.shader_program));
    
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        assert(get_gl_error(s_info.shader_program));

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, hit_data_ssbo);
        assert(get_gl_error(s_info.shader_program));
    pixel_data* pdata_ptr = (pixel_data*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(pixel_data) * img_width*img_height, GL_MAP_READ_BIT);
        assert(get_gl_error(s_info.shader_program));

    std::shared_ptr<std::vector<pixel_data>> pixel_data_vec = 
        std::make_shared<std::vector<pixel_data>>(std::vector<pixel_data>(img_width*img_height));

    int succesfull = 0;
    int not_found = 0;
    int uninit = 0;
    int d_val_zero = 0;

    int tr_count = 0;

    for (int i = 0; i < img_width*img_height; i++)
    {
        pixel_data_vec->push_back(pdata_ptr[i]);

        if (pdata_ptr[i].tr_i == -1)
            d_val_zero++;
        else if (pdata_ptr[i].tr_i == -2)
            not_found++;
        else if (pdata_ptr[i].tr_i == -3)
            uninit++;
        // DEBUG flag
        else if (pdata_ptr[i].tr_i == -4)
            std::cout << pdata_ptr[i].p_x << ", ";
        else
        {
            succesfull++;

            // if (pdata_ptr[i].padding0 < 0.7)
            //     continue;
            // std::cout << "dval: " << pdata_ptr[i].padding0 << ", dist:" << pdata_ptr[i].padding1 << "\n";
        }

    }

    std::cout << "\n";

    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        assert(get_gl_error(s_info.shader_program));

    glDeleteBuffers(1, &hit_data_ssbo);
        assert(get_gl_error(s_info.shader_program));
    glDeleteBuffers(1, &tr_ssbo);
        assert(get_gl_error(s_info.shader_program));

    // glDetachShader(s_info.shader_program, s_info.compute_shader);
    //     assert(get_gl_error(s_info.shader_program));
    glUseProgram(0);
        assert(get_gl_error(s_info.shader_program));
    // shader_cleanup();
    //     assert(get_gl_error(s_info.shader_program));

    std::cout << succesfull << " of points found, " <<  not_found << " of not found, " << d_val_zero << " of d_val_zero, " << uninit << " of uninitialized, " << d_val_zero + not_found + succesfull << " of total\n";

    delete[] pdata;

    return pixel_data_vec;
}


void shader_cleanup(shader_info s_info)
{
#ifdef OPEN3DGEN_HEADLESS
    PFNGLUSEPROGRAMPROC glUseProgram =  (PFNGLUSEPROGRAMPROC)eglGetProcAddress("glUseProgram");
    PFNGLDETACHSHADERPROC glDetachShader = (PFNGLDETACHSHADERPROC)eglGetProcAddress("glDetachShader");
 #endif
    glDetachShader(s_info.shader_program, s_info.compute_shader);
    glUseProgram(0);
 #ifdef OPEN3DGEN_HEADLESS
    eglTerminate(eglDpy);
 #else
    glfwDestroyWindow(s_info.window);
    glfwTerminate();
#endif
}

void setup_compute_shaders(shader_info& s_info)
{
    if (compute_shaders_init)
        return;
    
    compute_shaders_init = true;
#ifdef OPEN3DGEN_HEADLESS
/*
    EGLDisplay eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);

    EGLint major, minor;

    eglInitialize(eglDpy, &major, &minor);
*/
    PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT = reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(eglGetProcAddress("eglQueryDevicesEXT"));
    PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(eglGetProcAddress("eglGetPlatformDisplayEXT"));

    if (!eglQueryDevicesEXT || !eglGetPlatformDisplayEXT) {
        std::cerr << "Failed to get function pointers for EGL extensions" << std::endl;
            return;
    }

    EGLDisplay eglDpy;

    // Initialize EGL display
    EGLDeviceEXT device;
    EGLint numDevices;
    eglQueryDevicesEXT(0, NULL, &numDevices);
    EGLDeviceEXT* devices = new EGLDeviceEXT[numDevices];
    eglQueryDevicesEXT(numDevices, devices, &numDevices);
    device = devices[0];
    delete[] devices;

    EGLint attribs[] = { EGL_NONE };
    eglDpy = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, device, attribs);
    eglInitialize(eglDpy, NULL, NULL);

    // 2. Select an appropriate configuration
    EGLint numConfigs;
    EGLConfig eglCfg;

    eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);

    // 3. Create a surface
    EGLSurface eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg, 
                                            pbufferAttribs);

    // 4. Bind the API
    eglBindAPI(EGL_OPENGL_API);

    // 5. Create a context and make it current
    EGLint contextAttribs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 4,
        EGL_CONTEXT_MINOR_VERSION, 6,
        EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
        EGL_NONE
    };
    // 5. Create a context and make it current
    ctx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT,contextAttribs);

    eglMakeCurrent(eglDpy, NULL, NULL, ctx);

    PFNGLCREATEPROGRAMPROC glCreateProgram = (PFNGLCREATEPROGRAMPROC) eglGetProcAddress("glCreateProgram");
    PFNGLATTACHSHADERPROC glAttachShader = (PFNGLATTACHSHADERPROC) eglGetProcAddress("glAttachShader");
    PFNGLLINKPROGRAMPROC glLinkProgram = (PFNGLLINKPROGRAMPROC) eglGetProcAddress("glLinkProgram");
    PFNGLGETINTEGERI_VPROC glGetIntegeri_v = (PFNGLGETINTEGERI_VPROC) eglGetProcAddress("glGetIntegeri_v");


#else
    if (!glfwInit())
     {
        const char* description;
        int code = glfwGetError(&description);
        std::string desc = description;
        throw std::runtime_error("couldn't init GLFW, aborting in main " + desc);
     }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);

    s_info.window = glfwCreateWindow(640, 480, "stitcher3d", NULL, NULL);
    if (!s_info.window)
        throw std::runtime_error("couldn't create OpenGL window, aborting in main");

    glfwMakeContextCurrent(s_info.window);


    glewExperimental = GL_TRUE;
    glewInit();
#endif

    std::string shader_str = shader::read_shader(DATAFILE_PATH + TEXTURE_PROJECTION_COMPUTE_SHADER_PATH);

    // set the workgroup_size
    utilities::replace_string(shader_str, "layout (local_size_x = 1024) in;", "layout (local_size_x = " + std::to_string(s_info.workgroup_size) + ") in;");

    // set the camera parameters
    {
        // defaults in the shader
        // #define WIDTH 1280
        // #define HEIGHT 720
        // #define CX 637.74400638
        // #define CY 358.96757428
        // #define FX 912.91984103
        // #define FY 912.39513184

        utilities::replace_string(shader_str, "#define WIDTH 1280", "#define WIDTH " + std::to_string(s_info.img_width));
        utilities::replace_string(shader_str, "#define HEIGHT 720", "#define HEIGHT " + std::to_string(s_info.img_height));

        utilities::replace_string(shader_str, "#define CX 637.74400638", "#define CX " + std::to_string(s_info.intr_matrix.GetPrincipalPoint().first));
        utilities::replace_string(shader_str, "#define CY 358.96757428", "#define CY " + std::to_string(s_info.intr_matrix.GetPrincipalPoint().second));
        utilities::replace_string(shader_str, "#define FX 912.91984103", "#define FX " + std::to_string(s_info.intr_matrix.GetFocalLength().first));
        utilities::replace_string(shader_str, "#define FY 912.39513184", "#define FY " + std::to_string(s_info.intr_matrix.GetFocalLength().second));
    }


    GLuint cmp_shader = shader::compile_shader(shader_str);

    GLuint shader_program = glCreateProgram();
    assert (glGetError () == GL_NO_ERROR);

    glAttachShader(shader_program, cmp_shader);
        assert (glGetError () == GL_NO_ERROR);

    glLinkProgram(shader_program);
    assert (glGetError () == GL_NO_ERROR);

    {
        int wrk0 = 0;
        int wrk1 = 0;
        int wrk2 = 0;
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &wrk0);
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &wrk1);
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &wrk2);
        
        int work_grp_inv = 0;
        glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &work_grp_inv);

        std::cout << "work group info " << wrk0 << " " << wrk1 << " " << wrk2 << " " << work_grp_inv << " \n";
    }

    s_info.shader_program = shader_program;
    s_info.compute_shader = cmp_shader;
}

GLuint compile_shader(const std::string& shader)
{
    #ifdef OPEN3DGEN_HEADLESS
    PFNGLCREATESHADERPROC glCreateShader =  (PFNGLCREATESHADERPROC)eglGetProcAddress("glCreateShader");
    PFNGLSHADERSOURCEPROC glShaderSource = (PFNGLSHADERSOURCEPROC) eglGetProcAddress("glShaderSource");
    PFNGLCOMPILESHADERPROC glCompileShader = (PFNGLCOMPILESHADERPROC) eglGetProcAddress("glCompileShader");
    PFNGLGETSHADERIVPROC glGetShaderiv = (PFNGLGETSHADERIVPROC) eglGetProcAddress("glGetShaderiv");
    PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC) eglGetProcAddress("glGetShaderInfoLog");
    PFNGLDELETESHADERPROC glDeleteShader = (PFNGLDELETESHADERPROC) eglGetProcAddress("glDeleteShader");
    #endif
    GLuint shader_i = glCreateShader(GL_COMPUTE_SHADER);

    const char* shader_c_str = shader.c_str();

    glShaderSource(shader_i, 1, &shader_c_str, nullptr);
    glCompileShader(shader_i);
    //return shader_i;
    GLint compile_status = 0;
    glGetShaderiv(shader_i, GL_COMPILE_STATUS, &compile_status);
    if (compile_status != GL_FALSE)
        return shader_i;

    GLint log_size = 0;
    glGetShaderiv(shader_i, GL_INFO_LOG_LENGTH, &log_size);

    std::vector<GLchar> error_log(log_size);
    glGetShaderInfoLog(shader_i, log_size, &log_size, &error_log[0]);

    for (const GLchar& c : error_log)
        std::cout << c;

    std::cout << "\n";

    glDeleteShader(shader_i);

    throw std::runtime_error("couldn't compile shader, aborting in compile_shader\n");
    
    return 0;
}

const std::string read_shader(const std::string& shader_path)
{
    std::ifstream file(shader_path);

    if (!file.is_open())
        throw std::runtime_error("couldn't load shader " + shader_path + " aborting in read_shader\n");

    std::stringstream string_buffer;
    string_buffer << file.rdbuf();

    return string_buffer.str();
}


}
}
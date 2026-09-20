/*
    Minimal GLAD loader.

    The full generated glad.c defines a function pointer for every GL 3.3 core
    entry point. This project only touches a small subset, so instead of
    shipping ~1500 lines of generated code we define just the pointers we use.
    Unused pointers stay declared-but-undefined in glad.h; that is fine because
    nothing references them, so the linker never asks for them.

    Compile this as C (it matches the `extern "C"` declarations in glad.h):
        clang -c dependencies/src/glad.c -I dependencies/include -o glad.o
*/

#include <glad/glad.h>
#include <stddef.h>

struct gladGLversionStruct GLVersion = {0, 0};

/* --- buffers / vertex arrays ------------------------------------------- */
PFNGLGENVERTEXARRAYSPROC        glad_glGenVertexArrays        = NULL;
PFNGLBINDVERTEXARRAYPROC        glad_glBindVertexArray        = NULL;
PFNGLDELETEVERTEXARRAYSPROC     glad_glDeleteVertexArrays     = NULL;
PFNGLGENBUFFERSPROC             glad_glGenBuffers             = NULL;
PFNGLBINDBUFFERPROC             glad_glBindBuffer             = NULL;
PFNGLBUFFERDATAPROC             glad_glBufferData             = NULL;
PFNGLBUFFERSUBDATAPROC          glad_glBufferSubData          = NULL;
PFNGLDELETEBUFFERSPROC          glad_glDeleteBuffers          = NULL;

/* --- vertex attributes ------------------------------------------------- */
PFNGLVERTEXATTRIBPOINTERPROC    glad_glVertexAttribPointer    = NULL;
PFNGLENABLEVERTEXATTRIBARRAYPROC glad_glEnableVertexAttribArray = NULL;
PFNGLVERTEXATTRIBDIVISORPROC    glad_glVertexAttribDivisor    = NULL;

/* --- drawing / frame --------------------------------------------------- */
PFNGLDRAWELEMENTSINSTANCEDPROC  glad_glDrawElementsInstanced  = NULL;
PFNGLCLEARPROC                  glad_glClear                  = NULL;
PFNGLCLEARCOLORPROC             glad_glClearColor             = NULL;
PFNGLVIEWPORTPROC               glad_glViewport               = NULL;
PFNGLENABLEPROC                 glad_glEnable                 = NULL;
PFNGLBLENDFUNCPROC              glad_glBlendFunc              = NULL;
PFNGLGETSTRINGPROC              glad_glGetString              = NULL;

/* --- shader program ---------------------------------------------------- */
PFNGLCREATESHADERPROC           glad_glCreateShader           = NULL;
PFNGLSHADERSOURCEPROC           glad_glShaderSource           = NULL;
PFNGLCOMPILESHADERPROC          glad_glCompileShader          = NULL;
PFNGLGETSHADERIVPROC            glad_glGetShaderiv            = NULL;
PFNGLGETSHADERINFOLOGPROC       glad_glGetShaderInfoLog       = NULL;
PFNGLCREATEPROGRAMPROC          glad_glCreateProgram          = NULL;
PFNGLATTACHSHADERPROC           glad_glAttachShader           = NULL;
PFNGLLINKPROGRAMPROC            glad_glLinkProgram            = NULL;
PFNGLGETPROGRAMIVPROC           glad_glGetProgramiv           = NULL;
PFNGLGETPROGRAMINFOLOGPROC      glad_glGetProgramInfoLog      = NULL;
PFNGLDELETESHADERPROC           glad_glDeleteShader           = NULL;
PFNGLDELETEPROGRAMPROC          glad_glDeleteProgram          = NULL;
PFNGLUSEPROGRAMPROC             glad_glUseProgram             = NULL;
PFNGLGETUNIFORMLOCATIONPROC     glad_glGetUniformLocation     = NULL;
PFNGLUNIFORM1IPROC              glad_glUniform1i              = NULL;
PFNGLUNIFORM1FPROC              glad_glUniform1f              = NULL;
PFNGLUNIFORM2FPROC              glad_glUniform2f              = NULL;
PFNGLUNIFORM3FPROC              glad_glUniform3f              = NULL;

static void load_gl_pointers(GLADloadproc load)
{
    glad_glGenVertexArrays        = (PFNGLGENVERTEXARRAYSPROC)        load("glGenVertexArrays");
    glad_glBindVertexArray        = (PFNGLBINDVERTEXARRAYPROC)        load("glBindVertexArray");
    glad_glDeleteVertexArrays     = (PFNGLDELETEVERTEXARRAYSPROC)     load("glDeleteVertexArrays");
    glad_glGenBuffers             = (PFNGLGENBUFFERSPROC)             load("glGenBuffers");
    glad_glBindBuffer             = (PFNGLBINDBUFFERPROC)             load("glBindBuffer");
    glad_glBufferData             = (PFNGLBUFFERDATAPROC)             load("glBufferData");
    glad_glBufferSubData          = (PFNGLBUFFERSUBDATAPROC)          load("glBufferSubData");
    glad_glDeleteBuffers          = (PFNGLDELETEBUFFERSPROC)          load("glDeleteBuffers");

    glad_glVertexAttribPointer     = (PFNGLVERTEXATTRIBPOINTERPROC)    load("glVertexAttribPointer");
    glad_glEnableVertexAttribArray = (PFNGLENABLEVERTEXATTRIBARRAYPROC)load("glEnableVertexAttribArray");
    glad_glVertexAttribDivisor     = (PFNGLVERTEXATTRIBDIVISORPROC)    load("glVertexAttribDivisor");

    glad_glDrawElementsInstanced  = (PFNGLDRAWELEMENTSINSTANCEDPROC)  load("glDrawElementsInstanced");
    glad_glClear                  = (PFNGLCLEARPROC)                  load("glClear");
    glad_glClearColor             = (PFNGLCLEARCOLORPROC)             load("glClearColor");
    glad_glViewport               = (PFNGLVIEWPORTPROC)               load("glViewport");
    glad_glEnable                 = (PFNGLENABLEPROC)                 load("glEnable");
    glad_glBlendFunc              = (PFNGLBLENDFUNCPROC)              load("glBlendFunc");
    glad_glGetString              = (PFNGLGETSTRINGPROC)              load("glGetString");

    glad_glCreateShader           = (PFNGLCREATESHADERPROC)           load("glCreateShader");
    glad_glShaderSource           = (PFNGLSHADERSOURCEPROC)           load("glShaderSource");
    glad_glCompileShader          = (PFNGLCOMPILESHADERPROC)          load("glCompileShader");
    glad_glGetShaderiv            = (PFNGLGETSHADERIVPROC)            load("glGetShaderiv");
    glad_glGetShaderInfoLog       = (PFNGLGETSHADERINFOLOGPROC)       load("glGetShaderInfoLog");
    glad_glCreateProgram          = (PFNGLCREATEPROGRAMPROC)          load("glCreateProgram");
    glad_glAttachShader           = (PFNGLATTACHSHADERPROC)           load("glAttachShader");
    glad_glLinkProgram            = (PFNGLLINKPROGRAMPROC)            load("glLinkProgram");
    glad_glGetProgramiv           = (PFNGLGETPROGRAMIVPROC)           load("glGetProgramiv");
    glad_glGetProgramInfoLog      = (PFNGLGETPROGRAMINFOLOGPROC)      load("glGetProgramInfoLog");
    glad_glDeleteShader           = (PFNGLDELETESHADERPROC)           load("glDeleteShader");
    glad_glDeleteProgram          = (PFNGLDELETEPROGRAMPROC)          load("glDeleteProgram");
    glad_glUseProgram             = (PFNGLUSEPROGRAMPROC)             load("glUseProgram");
    glad_glGetUniformLocation     = (PFNGLGETUNIFORMLOCATIONPROC)     load("glGetUniformLocation");
    glad_glUniform1i              = (PFNGLUNIFORM1IPROC)              load("glUniform1i");
    glad_glUniform1f              = (PFNGLUNIFORM1FPROC)              load("glUniform1f");
    glad_glUniform2f              = (PFNGLUNIFORM2FPROC)              load("glUniform2f");
    glad_glUniform3f              = (PFNGLUNIFORM3FPROC)              load("glUniform3f");
}

int gladLoadGLLoader(GLADloadproc load)
{
    if (load == NULL)
        return 0;

    load_gl_pointers(load);

    /* We target the 3.3 core profile explicitly, so record that. */
    GLVersion.major = 3;
    GLVersion.minor = 3;

    /* Sanity check on one essential pointer. */
    return glad_glGenVertexArrays != NULL;
}

int gladLoadGL(void)
{
    /* This minimal loader has no platform default loader; callers should use
       gladLoadGLLoader with e.g. glfwGetProcAddress. */
    return 0;
}

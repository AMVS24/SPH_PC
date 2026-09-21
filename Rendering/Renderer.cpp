#include "Renderer.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>

#include "Square.h"
#include "dependencies/include/lib/shader_s.h"

namespace
{
    // Resize the GL viewport to follow the framebuffer (handles HiDPI + resizes).
    void framebufferSizeCallback(GLFWwindow *, int width, int height)
    {
        glViewport(0, 0, width, height);
    }
}

Renderer::Renderer(int width, int height, const char *title)
{
    if (!glfwInit())
    {
        std::cerr << "Renderer: failed to initialise GLFW\n";
        return;
    }

    // OpenGL 3.3 core profile.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Renderer: failed to create GLFW window\n";
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "Renderer: failed to load OpenGL via GLAD\n";
        glfwDestroyWindow(window);
        window = nullptr;
        glfwTerminate();
        return;
    }

    // Match the initial viewport to the real framebuffer size.
    int fbW, fbH;
    glfwGetFramebufferSize(window, &fbW, &fbH);
    glViewport(0, 0, fbW, fbH);

    shader = new Shader("shaders/particle.vs", "shaders/particle.fs");
    square = new Square();
}

Renderer::~Renderer()
{
    delete square;
    if (shader)
    {
        shader->del();
        delete shader;
    }
    if (window)
        glfwDestroyWindow(window);
    glfwTerminate();
}

void Renderer::setParticleColor(float r, float g, float b)
{
    color[0] = r;
    color[1] = g;
    color[2] = b;
}

bool Renderer::shouldClose() const
{
    return !window || glfwWindowShouldClose(window);
}

void Renderer::renderParticles(const vec2 *positions, int count, vec2 scale)
{
    if (!window || count <= 0)
        return;

    // Pack the double-precision positions into a tight float array for the GPU.
    instanceScratch.resize((size_t)count * 2);
    for (int i = 0; i < count; ++i)
    {
        instanceScratch[2 * i + 0] = (float)positions[i].x;
        instanceScratch[2 * i + 1] = (float)positions[i].y;
    }
    square->setInstanceData(instanceScratch.data(), count);

    glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    shader->use();
    glUniform2f(glGetUniformLocation(shader->ID, "uScale"), (float)scale.x, (float)scale.y);
    glUniform3f(glGetUniformLocation(shader->ID, "uColor"), color[0], color[1], color[2]);
    square->draw();

    glfwSwapBuffers(window);
    glfwPollEvents();
}

void Renderer::renderParticles(const std::vector<vec2> &positions, vec2 scale)
{
    renderParticles(positions.data(), (int)positions.size(), scale);
}

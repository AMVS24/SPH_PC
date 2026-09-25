#include "Renderer.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>

#include "Square.h"
#include "dependencies/include/lib/shader_s.h"

#define STB_IMAGE_IMPLEMENTATION
#include "dependencies/include/lib/stb_image.h"

namespace
{
    const char *PARTICLE_TEXTURE_PATH = "Media/circle_transparent.png";

    // Load an RGBA PNG (white circle, transparent background) into a GL texture.
    // Returns 0 on failure.
    unsigned int loadParticleTexture(const char *path)
    {
        stbi_set_flip_vertically_on_load(true); // match GL's bottom-left origin

        int width, height, channels;
        unsigned char *data = stbi_load(path, &width, &height, &channels, STBI_rgb_alpha);
        if (!data)
        {
            std::cerr << "Renderer: failed to load particle texture '" << path << "'\n";
            return 0;
        }

        unsigned int tex;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

        stbi_image_free(data);
        glBindTexture(GL_TEXTURE_2D, 0);
        return tex;
    }
}

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

    shader = new Shader("dependencies/shaders/particle.vs", "dependencies/shaders/particle.fs");
    square = new Square();
    particleTexture = loadParticleTexture(PARTICLE_TEXTURE_PATH);

    // Alpha blending so the texture's transparent background shows through
    // instead of drawing as an opaque square.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

Renderer::~Renderer()
{
    if (particleTexture)
        glDeleteTextures(1, &particleTexture);
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

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, particleTexture);
    glUniform1i(glGetUniformLocation(shader->ID, "uTex"), 0);
    glUniform1i(glGetUniformLocation(shader->ID, "uUseTexture"), useTexture_ ? 1 : 0);

    square->draw();

    glfwSwapBuffers(window);
    glfwPollEvents();
}

void Renderer::renderParticles(const std::vector<vec2> &positions, vec2 scale)
{
    renderParticles(positions.data(), (int)positions.size(), scale);
}

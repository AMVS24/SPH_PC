#ifndef RENDERER_H
#define RENDERER_H

#include <vector>

#include "../sequential_c/backend/linalg.h" // vec2

// Forward declarations keep GLFW/shader headers out of this interface.
struct GLFWwindow;
class Shader;
class Square;

// Owns a GLFW window + OpenGL 3.3 core context and knows how to draw a cloud of
// square particles. Typical use:
//
//     Renderer renderer(800, 600, "SPH");
//     while (!renderer.shouldClose())
//         renderer.renderParticles(positions, count, {0.02, 0.02});
//
class Renderer
{
public:
    Renderer(int width, int height, const char *title);
    ~Renderer();

    Renderer(const Renderer &) = delete;
    Renderer &operator=(const Renderer &) = delete;

    // Draw one frame: clears the screen, draws a scaled square at every
    // position, then presents the frame and pumps window events. `scale` is the
    // half-extent of each square on the x and y axes (in NDC units).
    void renderParticles(const vec2 *positions, int count, vec2 scale);
    void renderParticles(const std::vector<vec2> &positions, vec2 scale);

    // True once the user has asked to close the window.
    bool shouldClose() const;

    // Solid fill colour for the squares (RGB, 0..1). Defaults to white.
    void setParticleColor(float r, float g, float b);

    bool ok() const { return window != nullptr; }

private:
    GLFWwindow *window = nullptr;
    Shader *shader = nullptr;
    Square *square = nullptr;

    // Reused scratch buffer: vec2 (double) positions packed into floats for GL.
    std::vector<float> instanceScratch;

    float color[3] = {1.0f, 1.0f, 1.0f};
};

#endif

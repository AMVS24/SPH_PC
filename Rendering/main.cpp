#include <cmath>
#include <vector>

#include "Renderer.h"

// Minimal demo: scatter a grid of squares and let them drift, to exercise the
// Renderer + Square instancing path.
int main()
{
    Renderer renderer(800, 800, "SPH particles");
    if (!renderer.ok())
        return 1;

    renderer.setParticleColor(0.35f, 0.65f, 1.0f);

    // A grid of positions in normalised device coordinates [-1, 1].
    std::vector<vec2> positions;
    const int side = 20;
    for (int i = 0; i < side; ++i)
        for (int j = 0; j < side; ++j)
        {
            double x = -0.8 + 1.6 * i / (side - 1);
            double y = -0.8 + 1.6 * j / (side - 1);
            positions.push_back({x, y});
        }

    const vec2 scale = {0.03, 0.03}; // half-extent of each square, in NDC

    double t = 0.0;
    while (!renderer.shouldClose())
    {
        // Wobble the whole cloud so we can see it animate.
        std::vector<vec2> frame = positions;
        for (auto &p : frame)
            p.x += 0.03 * std::sin(t + p.y * 4.0);
        t += 0.02;

        renderer.renderParticles(frame, scale);
    }
    return 0;
}

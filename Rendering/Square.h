#ifndef SQUARE_H
#define SQUARE_H

#include <glad/glad.h>

// A single unit square (centred on the origin, corners at +/-0.5) that is drawn
// many times in one call via instancing. Each instance is offset by a per-particle
// position supplied through setInstanceData(); a uniform scale is applied in the
// vertex shader. One Square owns all the GL objects needed to draw an arbitrary
// number of particles.
class Square
{
public:
    Square()
    {
        // Unit quad centred on origin: the vertex shader offsets and scales it.
        const float vertices[] = {
            -0.5f, -0.5f,
             0.5f, -0.5f,
             0.5f,  0.5f,
            -0.5f,  0.5f,
        };
        const unsigned int indices[] = {
            0, 1, 2,
            2, 3, 0,
        };

        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &quadVBO);
        glGenBuffers(1, &EBO);
        glGenBuffers(1, &instanceVBO);

        glBindVertexArray(VAO);

        // location 0: the static quad geometry, shared by every instance.
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

        // location 1: per-instance offset. Divisor 1 => advances once per instance.
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
        glVertexAttribDivisor(1, 1);

        glBindVertexArray(0);
    }

    ~Square()
    {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &quadVBO);
        glDeleteBuffers(1, &EBO);
        glDeleteBuffers(1, &instanceVBO);
    }

    // Non-copyable: it owns raw GL handles.
    Square(const Square &) = delete;
    Square &operator=(const Square &) = delete;

    // Upload `count` instance offsets, tightly packed as [x0,y0, x1,y1, ...].
    // The buffer grows on demand and is reused across frames otherwise.
    void setInstanceData(const float *xy, int count)
    {
        const GLsizeiptr bytes = (GLsizeiptr)count * 2 * sizeof(float);
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        if (count > capacity)
        {
            glBufferData(GL_ARRAY_BUFFER, bytes, xy, GL_DYNAMIC_DRAW);
            capacity = count;
        }
        else
        {
            glBufferSubData(GL_ARRAY_BUFFER, 0, bytes, xy);
        }
        instanceCount = count;
    }

    // Draw every uploaded instance in a single call.
    void draw() const
    {
        glBindVertexArray(VAO);
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, instanceCount);
        glBindVertexArray(0);
    }

private:
    unsigned int VAO = 0;
    unsigned int quadVBO = 0;
    unsigned int EBO = 0;
    unsigned int instanceVBO = 0;

    int capacity = 0;      // instances the instanceVBO can currently hold
    int instanceCount = 0; // instances uploaded for the current frame
};

#endif

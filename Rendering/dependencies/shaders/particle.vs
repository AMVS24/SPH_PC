#version 330 core

// Unit-square corner, in [-0.5, 0.5] on both axes.
layout (location = 0) in vec2 aPos;
// Per-instance particle centre, in normalised device coordinates [-1, 1].
layout (location = 1) in vec2 aOffset;

// Half-extents applied to every square (x and y independently).
uniform vec2 uScale;

void main()
{
    vec2 ndc = aPos * uScale + aOffset;
    gl_Position = vec4(ndc, 0.0, 1.0);
}

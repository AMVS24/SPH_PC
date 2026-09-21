#version 330 core

out vec4 FragColor;
in vec3 vertexColor;
in vec2 TexCoord;

uniform sampler2D imageTexture1;
uniform sampler2D imageTexture2;

uniform float opacity;

// Parameters for the LCG
const int a = 1664525;
const int c = 1013904223;
const int m = 4294967295; // 2^32 - 1

const float repeat = 2.0;

float rand(vec2 seed) {
    return fract(sin(dot(seed.xy, vec2(0.02898, 0.0233))) * 43758.5453);
}

void main()
{
    // FragColor = texture(imageTexture1, TexCoord)*0.5 + vec4(vertexColor, 1.0) * rand(gl_FragCoord.xy);
    //* TexCoord has to be from 0-1 (hence 1.0-TexCoord)
    FragColor = mix(texture(imageTexture1, TexCoord*repeat), texture(imageTexture2, TexCoord*repeat), opacity)*0.5 + vec4(vertexColor, 1.0) * rand(gl_FragCoord.xy);
} 
#version 330 core

in vec2 vTexCoord;
out vec4 FragColor;

uniform vec3 uColor;
uniform sampler2D uTex;
// When true, sample uTex (circle, tinted by uColor, texture's own alpha).
// When false, draw a plain opaque square of uColor -- the pre-texture look.
uniform bool uUseTexture;

void main()
{
    vec4 tex = uUseTexture ? texture(uTex, vTexCoord) : vec4(1.0);
    FragColor = vec4(uColor * tex.rgb, tex.a);
}

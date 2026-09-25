#include <constants.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "../Rendering/Renderer.h"

using namespace std;

// This project's glad.c is hand-trimmed to only the GL calls Renderer.cpp
// already uses, so glReadPixels/glReadBuffer/glPixelStorei aren't defined
// there. Rather than edit the shared Rendering/ glad build, define + load
// just these three ourselves via the same glfwGetProcAddress loader glad
// itself uses -- self-contained to this file.
PFNGLPIXELSTOREIPROC glad_glPixelStorei = NULL;
PFNGLREADBUFFERPROC glad_glReadBuffer = NULL;
PFNGLREADPIXELSPROC glad_glReadPixels = NULL;

// Streams frames straight from positions.bin (read-only, never seeks
// backward, never loads more than one frame at a time -- safe for a
// multi-GB file), renders each with the existing Renderer, reads the
// framebuffer back, and pipes raw RGB straight into ffmpeg's stdin.
int main(int argc, char** argv){
    const char* in_path = argc > 1 ? argv[1] : "positions.bin";
    const char* out_path = argc > 2 ? argv[2] : "replay.mp4";
    int fps = argc > 3 ? atoi(argv[3]) : 60;
    int speed = argc > 4 ? atoi(argv[4]) : 1;

    FILE* f = fopen(in_path, "rb");
    if(!f){
        printf("could not open %s\n", in_path);
        return 1;
    }
    int n;
    fread(&n, sizeof(int), 1, f);

    // popen() succeeds as soon as it can spawn a cmd.exe host, even if the
    // command inside it (ffmpeg) doesn't exist on PATH -- cmd.exe just prints
    // "not recognized" and exits, and silently writing frames into that dead
    // pipe would otherwise look like a successful run. Fail fast instead.
    if(system("ffmpeg -version >NUL 2>&1") != 0){
        printf("ffmpeg not found on PATH -- install it or open a fresh terminal if you just installed it\n");
        fclose(f);
        return 1;
    }

    vector<vec2> frame(n);
    vector<vec2> ndc(n);

    const int width = 1000, height = 1000;
    Renderer renderer(width, height, "SPH video render");
    renderer.setParticleColor(0.0f, 0.0f, 1.0f);

    glad_glPixelStorei = (PFNGLPIXELSTOREIPROC)glfwGetProcAddress("glPixelStorei");
    glad_glReadBuffer = (PFNGLREADBUFFERPROC)glfwGetProcAddress("glReadBuffer");
    glad_glReadPixels = (PFNGLREADPIXELSPROC)glfwGetProcAddress("glReadPixels");

    const double sx = 2.0/(WORLD_MAX_X - WORLD_MIN_X);
    const double sy = 2.0/(WORLD_MAX_Y - WORLD_MIN_Y);
    const double scale_xy = PARTICLE_SIZE_TO_H*H*PARTICLE_NDC_PER_WORLD_UNIT;

    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size %dx%d -framerate %d -i - -vf vflip -c:v libx264 -pix_fmt yuv420p \"%s\"",
        width, height, fps, out_path);
    FILE* ffmpeg = popen(cmd, "wb");
    if(!ffmpeg){
        printf("could not launch ffmpeg -- is it on PATH?\n");
        return 1;
    }

    vector<unsigned char> pixels((size_t)width*height*3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    long long frame_idx = 0, written = 0;
    while(fread(frame.data(), sizeof(vec2), n, f) == (size_t)n){
        if(renderer.shouldClose()) break;

        if(frame_idx % speed == 0){
            for(int i = 0; i < n; i++){
                ndc[i].x = (frame[i].x - WORLD_MIN_X)*sx - 1.0;
                ndc[i].y = (frame[i].y - WORLD_MIN_Y)*sy - 1.0;
            }
            renderer.renderParticles(ndc.data(), n, vec2{scale_xy, scale_xy});

            glReadBuffer(GL_FRONT);
            glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
            fwrite(pixels.data(), 1, pixels.size(), ffmpeg);

            written++;
            if(written % 300 == 0){
                printf("\rencoded %lld frames", written);
                fflush(stdout);
            }
        }
        frame_idx++;
    }
    fclose(f);
    int ffmpeg_status = pclose(ffmpeg);
    if(ffmpeg_status != 0){
        printf("\nffmpeg exited with an error (status %d) -- %s is probably incomplete or missing\n", ffmpeg_status, out_path);
        return 1;
    }
    printf("\nwrote %lld frames to %s\n", written, out_path);
}

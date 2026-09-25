#include <constants.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <chrono>
#include <thread>
#include "../Rendering/Renderer.h"

using namespace std;

int main(int argc, char** argv){
    const char* in_path = argc > 1 ? argv[1] : "positions.bin";
    int speed = argc > 2 ? atoi(argv[2]) : 1;
    const int fps = 60;

    FILE* f = fopen(in_path, "rb");
    if(!f){
        printf("could not open %s\n", in_path);
        return 1;
    }
    int n;
    fread(&n, sizeof(int), 1, f);

    vector<vector<vec2>> frames;
    vector<vec2> frame(n);
    while(fread(frame.data(), sizeof(vec2), n, f) == (size_t)n) frames.push_back(frame);
    fclose(f);
    printf("loaded %d timesteps of %d particles\n", (int)frames.size(), n);
    if(frames.empty()) return 1;

    Renderer renderer(1000, 1000, "SPH replay");
    renderer.setParticleColor(0.0f, 0.0f, 1.0f);

    const double sx = 2.0/(WORLD_MAX_X - WORLD_MIN_X);
    const double sy = 2.0/(WORLD_MAX_Y - WORLD_MIN_Y);
    const double scale_xy = PARTICLE_SIZE_TO_H*H*PARTICLE_NDC_PER_WORLD_UNIT;
    vector<vec2> ndc(n);
    auto frame_time = chrono::microseconds(1000000/fps);

    size_t k = 0;
    while(!renderer.shouldClose()){
        auto t0 = chrono::steady_clock::now();
        for(int i = 0; i < n; i++){
            ndc[i].x = (frames[k][i].x - WORLD_MIN_X)*sx - 1.0;
            ndc[i].y = (frames[k][i].y - WORLD_MIN_Y)*sy - 1.0;
        }
        renderer.renderParticles(ndc.data(), n, vec2{scale_xy, scale_xy});
        k = min(k + speed, frames.size() - 1);
        this_thread::sleep_until(t0 + frame_time);
    }
}

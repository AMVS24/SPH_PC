#include <physics.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include "profiler/Profiler.h"

struct SceneParams {
    double dx = 0;
    double mass = 0;
    double rho0 = 0;
    double g = 0;
    double domain[4] = {0,0,0,0};
    double t_end = 0;
    double snapshot_dt = 0;
    double fill_height = -1;
    double column_height = -1;
    bool has_fill_height = false;
    bool has_column_height = false;
};

static std::string trim(const std::string& s){
    size_t a = s.find_first_not_of(" \t\r\n");
    if(a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b-a+1);
}

static SceneParams load_scene_json(const std::string& path){
    std::ifstream f(path);
    if(!f){
        fprintf(stderr, "cannot open %s\n", path.c_str());
        exit(1);
    }
    SceneParams p;
    std::string line;
    std::string pending_key;
    std::vector<double> pending_array;
    bool in_array = false;

    while(std::getline(f, line)){
        std::string t = trim(line);
        if(t.empty() || t == "{" || t == "}") continue;

        if(in_array){
            if(t[0] == ']'){
                if(pending_key == "domain" && pending_array.size() == 4){
                    for(int i=0;i<4;i++) p.domain[i] = pending_array[i];
                }
                in_array = false;
                pending_array.clear();
                continue;
            }
            pending_array.push_back(std::stod(t));
            continue;
        }

        size_t colon = t.find(':');
        if(colon == std::string::npos) continue;
        std::string key = trim(t.substr(0, colon));
        std::string val = trim(t.substr(colon+1));
        if(!val.empty() && val.back() == ',') val.pop_back();

        if(key.size() >= 2 && key.front() == '"' && key.back() == '"')
            key = key.substr(1, key.size()-2);

        if(!val.empty() && val[0] == '['){
            in_array = true;
            pending_key = key;
            pending_array.clear();
            continue;
        }
        if(val.empty() || val == "\"\"") continue;

        double num;
        try { num = std::stod(val); } catch(...) { continue; }

        if(key == "dx") p.dx = num;
        else if(key == "mass") p.mass = num;
        else if(key == "rho0") p.rho0 = num;
        else if(key == "g") p.g = num;
        else if(key == "t_end") p.t_end = num;
        else if(key == "snapshot_dt") p.snapshot_dt = num;
        else if(key == "fill_height"){ p.fill_height = num; p.has_fill_height = true; }
        else if(key == "column_height"){ p.column_height = num; p.has_column_height = true; }
    }
    return p;
}

static void load_init_csv(const std::string& path, std::vector<vec2>& pos, std::vector<vec2>& vel){
    std::ifstream f(path);
    if(!f){
        fprintf(stderr, "cannot open %s\n", path.c_str());
        exit(1);
    }
    std::string line;
    std::getline(f, line); // header
    while(std::getline(f, line)){
        std::string t = trim(line);
        if(t.empty()) continue;
        std::stringstream ss(t);
        std::string tok;
        double v[4];
        for(int i=0;i<4;i++){
            std::getline(ss, tok, ',');
            v[i] = std::stod(tok);
        }
        pos.push_back(vec2{v[0], v[1]});
        vel.push_back(vec2{v[2], v[3]});
    }
}

static void enforce_domain_boundary(vec2* position_array, vec2* velocity_array, int n, const double domain[4]){
    double min_x = domain[0], min_y = domain[1], max_x = domain[2], max_y = domain[3];
    for(int i = 0; i<n; i++){
        if(position_array[i].x < min_x){
            position_array[i].x = min_x;
            velocity_array[i].x = -velocity_array[i].x*WALL_RESTITUTION;
        }
        else if(position_array[i].x > max_x){
            position_array[i].x = max_x;
            velocity_array[i].x = -velocity_array[i].x*WALL_RESTITUTION;
        }
        if(position_array[i].y < min_y){
            position_array[i].y = min_y;
            velocity_array[i].y = -velocity_array[i].y*WALL_RESTITUTION;
        }
        else if(position_array[i].y > max_y){
            position_array[i].y = max_y;
            velocity_array[i].y = -velocity_array[i].y*WALL_RESTITUTION;
        }
    }
}

int main(int argc, char** argv){
    if(argc < 5){
        fprintf(stderr, "usage: %s scene.json scene.init.csv out.csv out.steps.csv\n", argv[0]);
        return 1;
    }
    std::string json_path = argv[1];
    std::string init_path = argv[2];
    std::string out_path = argv[3];
    std::string steps_path = argv[4];

    g_profiler.begin("Initialisation");

    SceneParams sp = load_scene_json(json_path);

    std::vector<vec2> pos, vel;
    load_init_csv(init_path, pos, vel);
    int n_fluid = (int)pos.size();

    H = 1.5*sp.dx;
    PARTICLE_MASS = sp.mass;
    TARGET_DENSITY = sp.rho0;
    GRAVITY = sp.g;

    double v_char;
    if(sp.g > 0){
        double L = sp.has_fill_height ? sp.fill_height : (sp.has_column_height ? sp.column_height : (sp.domain[3]-sp.domain[1]));
        v_char = std::sqrt(2.0*sp.g*L);
    } else {
        v_char = 0.0;
        for(int i=0;i<n_fluid;i++) v_char = std::max(v_char, vnorm(vel[i]));
    }
    double c0 = 10.0*v_char;
    if(c0 < 1.0) c0 = 1.0;
    PRESSURE_MULTIPLIER = c0*c0;

    int n = n_fluid;

    vec2* position_array = new vec2[n];
    vec2* velocity_array = new vec2[n];
    vec2* acceleration_array = new vec2[n];
    for(int i=0;i<n_fluid;i++){
        position_array[i] = pos[i];
        velocity_array[i] = vel[i];
        acceleration_array[i] = vec2{0,0};
    }

    vecN density_array(n);
    vecN pressure_array(n);

    g_profiler.end("Initialisation");

    std::ofstream out(out_path);
    out << std::setprecision(17);
    out << "frame,t,id,x,y,vx,vy,rho,p\n";
    std::ofstream steps(steps_path);
    steps << std::setprecision(17);
    steps << "step,t,dt,h,c0,amax,vmax\n";

    update_forces(position_array, velocity_array, acceleration_array, density_array, pressure_array, n, n_fluid);

    int frame = 0;
    double t = 0.0;
    double next_snapshot = 0.0;

    auto write_snapshot = [&](){
        for(int i=0;i<n_fluid;i++){
            out << frame << "," << t << "," << i << ","
                << position_array[i].x << "," << position_array[i].y << ","
                << velocity_array[i].x << "," << velocity_array[i].y << ","
                << density_array[i] << "," << pressure_array[i] << "\n";
        }
        frame++;
        next_snapshot += sp.snapshot_dt;
    };

    write_snapshot();

    const double C = 0.3;
    long long step = 0;

    while(t < sp.t_end){
        double vmax = 0, amax = 0;
        for(int i=0;i<n_fluid;i++){
            vmax = std::max(vmax, vnorm(velocity_array[i]));
            amax = std::max(amax, vnorm(acceleration_array[i]));
        }
        if(amax <= 0) amax = 1e-9;

        double dt_cfl = C*H/(c0+vmax);
        double dt_force = C*std::sqrt(H/amax);
        double dt = std::min(dt_cfl, dt_force);
        if(t + dt > sp.t_end) dt = sp.t_end - t;
        if(dt <= 0) break;

        steps << step << "," << t << "," << dt << "," << H << "," << c0 << "," << amax << "," << vmax << "\n";
        if(step % 500 == 0){
            fprintf(stderr, "step=%lld t=%.6f dt=%.9f amax=%.6f vmax=%.6f\n", step, t, dt, amax, vmax);
        }

        {
            ScopedTimer t_integ(g_profiler, "Integration");
            for(int i=0;i<n_fluid;i++) velocity_array[i] += acceleration_array[i]*0.5*dt;
            for(int i=0;i<n_fluid;i++) position_array[i] += velocity_array[i]*dt;
            enforce_domain_boundary(position_array, velocity_array, n_fluid, sp.domain);
        }
        update_forces(position_array, velocity_array, acceleration_array, density_array, pressure_array, n, n_fluid);
        {
            ScopedTimer t_integ(g_profiler, "Integration");
            for(int i=0;i<n_fluid;i++) velocity_array[i] += acceleration_array[i]*0.5*dt;
        }

        t += dt;
        step++;

        if(t >= next_snapshot - 1e-12 || t >= sp.t_end - 1e-12){
            write_snapshot();
        }
    }

    g_profiler.report(std::cerr);
    printf("done: n_fluid=%d steps=%lld t=%.6f frames=%d\n", n_fluid, step, t, frame);
    return 0;
}

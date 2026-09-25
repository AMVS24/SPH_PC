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
#include <chrono>
#include <omp.h>

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

static double ms_since(std::chrono::high_resolution_clock::time_point t0){
    return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
}

struct Nbr { int j; double norm; vec2 diff; };

static void update_forces_breakdown(vec2* position_array, vec2* velocity_array, vec2* acceleration_array, vecN& density_array, vecN& pressure_array, int n, int n_fluid, std::vector<std::vector<Nbr>>& nb, double stage_ms[6]){
    auto t0 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i<n; i++){
        nb[i].clear();
        for(int j = 0; j<n; j++){
            vec2 diff = position_array[j]-position_array[i];
            double norm = vnorm(diff);
            if(norm < 2*H) nb[i].push_back({j, norm, diff});
        }
    }
    stage_ms[0] += ms_since(t0);

    t0 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i<n; i++){
        density_array[i] = 0;
        for(const Nbr& k : nb[i]) density_array[i] += PARTICLE_MASS*poly_6_kernel(k.norm/H);
    }
    stage_ms[1] += ms_since(t0);

    t0 = std::chrono::high_resolution_clock::now();
    pressure_array = (density_array-TARGET_DENSITY)*PRESSURE_MULTIPLIER;
    stage_ms[4] += ms_since(t0);

    double c0 = sqrt(PRESSURE_MULTIPLIER);

    t0 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i<n_fluid; i++){
        nb[i].clear();
        for(int j = 0; j<n; j++){
            if(i != j && density_array[j] != 0){
                vec2 diff = position_array[j]-position_array[i];
                double norm = vnorm(diff);
                if(norm != 0 && norm < 2*H) nb[i].push_back({j, norm, diff});
            }
        }
    }
    stage_ms[2] += ms_since(t0);

    t0 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i<n_fluid; i++){
        acceleration_array[i] = vec2{0,0};
        for(const Nbr& k : nb[i]){
            int j = k.j;
            vec2 grad = spiky_kernel(k.norm/H, k.diff/k.norm);
            acceleration_array[i] -= (pressure_array[i]/(density_array[i]*density_array[i]) + pressure_array[j]/(density_array[j]*density_array[j]))*grad;
            double dot_prod = dot(k.diff, velocity_array[j] - velocity_array[i]);
            if(dot_prod < 0){
                double mu = H*dot_prod/(k.norm*k.norm + 0.0001*H*H);
                double avg_density = (density_array[i] + density_array[j])*0.5;
                double pi = (-VISCOSITY_ALPHA*c0*mu)/(avg_density + 1e-9);
                acceleration_array[i] -= pi*grad;
            }
        }
        acceleration_array[i].y -= GRAVITY;
    }
    stage_ms[3] += ms_since(t0);
}

int main(int argc, char** argv){
    if(argc < 5){
        fprintf(stderr, "usage: %s scene.json scene.init.csv out.csv out.steps.csv [--max-steps N] [--breakdown] [--threads N]\n", argv[0]);
        return 1;
    }
    std::string json_path = argv[1];
    std::string init_path = argv[2];
    std::string out_path = argv[3];
    std::string steps_path = argv[4];

    long long max_steps = -1;
    bool breakdown = false;
    for(int a=5; a<argc; a++){
        std::string arg = argv[a];
        if(arg == "--max-steps" && a+1 < argc) max_steps = std::stoll(argv[++a]);
        else if(arg == "--breakdown") breakdown = true;
        else if(arg == "--threads" && a+1 < argc) omp_set_num_threads(std::atoi(argv[++a]));
    }

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

    const int WINDOW = 10;
    std::vector<double> window_ms_per_step;
    double window_accum_ms = 0;
    int window_count = 0;

    std::vector<std::vector<Nbr>> nb(breakdown ? n : 0);
    double stage_accum[6] = {0, 0, 0, 0, 0, 0};
    std::vector<double> stage_windows[6];

    while(t < sp.t_end){
        if(max_steps >= 0 && step >= max_steps) break;
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

        auto step_t0 = std::chrono::high_resolution_clock::now();

        for(int i=0;i<n_fluid;i++) velocity_array[i] += acceleration_array[i]*0.5*dt;
        for(int i=0;i<n_fluid;i++) position_array[i] += velocity_array[i]*dt;
        auto bc_t0 = std::chrono::high_resolution_clock::now();
        enforce_domain_boundary(position_array, velocity_array, n_fluid, sp.domain);
        if(breakdown){
            stage_accum[5] += ms_since(bc_t0);
            update_forces_breakdown(position_array, velocity_array, acceleration_array, density_array, pressure_array, n, n_fluid, nb, stage_accum);
        }
        else{
            update_forces(position_array, velocity_array, acceleration_array, density_array, pressure_array, n, n_fluid);
        }
        for(int i=0;i<n_fluid;i++) velocity_array[i] += acceleration_array[i]*0.5*dt;

        double step_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - step_t0).count();
        window_accum_ms += step_ms;
        window_count++;
        if(window_count == WINDOW){
            window_ms_per_step.push_back(window_accum_ms / WINDOW);
            window_accum_ms = 0;
            window_count = 0;
            for(int s=0;s<6;s++){
                stage_windows[s].push_back(stage_accum[s] / WINDOW);
                stage_accum[s] = 0;
            }
        }

        t += dt;
        step++;

        if(t >= next_snapshot - 1e-12 || t >= sp.t_end - 1e-12){
            write_snapshot();
        }
    }

    double window_mean = 0, window_stdev = 0;
    if(!window_ms_per_step.empty()){
        for(double v : window_ms_per_step) window_mean += v;
        window_mean /= window_ms_per_step.size();
        for(double v : window_ms_per_step) window_stdev += (v-window_mean)*(v-window_mean);
        window_stdev = window_ms_per_step.size() > 1 ? std::sqrt(window_stdev/(window_ms_per_step.size()-1)) : 0.0;
    }

    printf("done: n_fluid=%d steps=%lld t=%.6f frames=%d threads=%d\n", n_fluid, step, t, frame, omp_get_max_threads());
    printf("window_stats: window_size=%d n_windows=%zu mean_ms_per_step=%.6f stdev_ms_per_step=%.6f\n",
           WINDOW, window_ms_per_step.size(), window_mean, window_stdev);
    if(breakdown){
        const char* stage_names[6] = {"A", "B", "C", "D", "E", "F"};
        printf("breakdown:");
        for(int s=0;s<6;s++){
            double m = 0, sd = 0;
            for(double v : stage_windows[s]) m += v;
            if(!stage_windows[s].empty()) m /= stage_windows[s].size();
            for(double v : stage_windows[s]) sd += (v-m)*(v-m);
            sd = stage_windows[s].size() > 1 ? std::sqrt(sd/(stage_windows[s].size()-1)) : 0.0;
            printf(" %s_mean=%.6f %s_stdev=%.6f", stage_names[s], m, stage_names[s], sd);
        }
        printf("\n");
    }
    return 0;
}

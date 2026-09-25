#include "Profiler.h"

#include <fstream>
#include <iomanip>

Profiler g_profiler;

void Profiler::begin(const std::string& name){
    active_[name] = std::chrono::high_resolution_clock::now();
}

void Profiler::end(const std::string& name){
    auto it = active_.find(name);
    if(it == active_.end()) return; // end() without a matching begin() -- ignore

    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - it->second).count();

    Timing& t = totals_[name];
    t.total_ms += ms;
    t.calls += 1;
}

void Profiler::add(const std::string& name, double ms, long long calls){
    Timing& t = totals_[name];
    t.total_ms += ms;
    t.calls += calls;
}

void Profiler::report(std::ostream& out) const {
    double grand_total = 0.0;
    for(const auto& entry : totals_) grand_total += entry.second.total_ms;

    out << "\n--- Profiler report ---\n";
    out << std::left << std::setw(20) << "Section"
        << std::right << std::setw(14) << "Total ms"
        << std::setw(12) << "Calls"
        << std::setw(16) << "Avg ms/call"
        << std::setw(10) << "% time" << "\n";

    for(const auto& entry : totals_){
        const std::string& name = entry.first;
        const Timing& t = entry.second;
        double avg = t.calls ? t.total_ms / t.calls : 0.0;
        double pct = grand_total > 0.0 ? 100.0 * t.total_ms / grand_total : 0.0;

        out << std::left << std::setw(20) << name
            << std::right << std::setw(14) << std::fixed << std::setprecision(3) << t.total_ms
            << std::setw(12) << t.calls
            << std::setw(16) << std::fixed << std::setprecision(5) << avg
            << std::setw(9) << std::fixed << std::setprecision(1) << pct << "%\n";
    }

    out << std::left << std::setw(20) << "Total"
        << std::right << std::setw(14) << std::fixed << std::setprecision(3) << grand_total << " ms\n";
    out << "-----------------------\n";
    out.flush(); // stdout is fully-buffered (not line-buffered) once redirected to a
                 // file/pipe, so without this the report can sit unseen until exit.
}

void Profiler::write_csv(const std::string& path) const {
    std::ofstream f(path);
    if(!f) return;

    double grand_total = 0.0;
    for(const auto& entry : totals_) grand_total += entry.second.total_ms;

    f << "section,total_ms,calls,avg_ms,percent\n";
    for(const auto& entry : totals_){
        const std::string& name = entry.first;
        const Timing& t = entry.second;
        double avg = t.calls ? t.total_ms / t.calls : 0.0;
        double pct = grand_total > 0.0 ? 100.0 * t.total_ms / grand_total : 0.0;
        f << name << "," << t.total_ms << "," << t.calls << "," << avg << "," << pct << "\n";
    }
}

void Profiler::reset(){
    active_.clear();
    totals_.clear();
}

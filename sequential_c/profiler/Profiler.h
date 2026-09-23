#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <iostream>

// Lightweight section-timing profiler. Sections are named at the call site
// (no fixed enum), so new boundaries can be added just by wrapping more code
// in a ScopedTimer -- nothing here needs to change.
//
// Usage:
//     {
//         ScopedTimer t(g_profiler, "Density Compute");
//         ... code to time ...
//     }   // timer stops automatically when this scope ends
//
// Multiple ScopedTimer calls with the same name (e.g. across many substeps,
// or two separate blocks within one function) accumulate into the same
// running total, so report() shows the overall time and share for that
// section across the whole run.
class Profiler {
public:
    void begin(const std::string& name);
    void end(const std::string& name);
    void add(const std::string& name, double ms, long long calls = 1);

    // Prints a table of total ms, call count, avg ms/call and % of total
    // time across every named section seen so far.
    void report(std::ostream& out = std::cout) const;

    // Writes the same data as CSV (section,total_ms,calls,avg_ms,percent).
    void write_csv(const std::string& path) const;

    void reset();

private:
    struct Timing {
        double total_ms = 0.0;
        long long calls = 0;
    };

    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> active_;
    std::unordered_map<std::string, Timing> totals_;
};

// RAII wrapper: begin()s on construction, end()s on destruction (including
// on an early return), so call sites never need to remember to stop a timer.
class ScopedTimer {
public:
    ScopedTimer(Profiler& profiler, std::string name)
        : profiler_(profiler), name_(std::move(name)) {
        profiler_.begin(name_);
    }
    ~ScopedTimer() {
        profiler_.end(name_);
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    Profiler& profiler_;
    std::string name_;
};

// One shared instance for the whole program -- mirrors the plain global
// `profiler` object in sequential_py/naive_py.py's Profiler class, so the
// same section names/usage pattern carries over from the Python baseline.
extern Profiler g_profiler;

#pragma once

#include <iostream>
#include <chrono>
#include <string>
#include <vector>


namespace stitcher3d
{

class Timer
{
public:
    Timer() :
        m_times({ std::chrono::high_resolution_clock::now() })
    { }

    ~Timer() {}

    void stop(const std::string& msg = "task", const bool leading_space = false)
    {
        const auto stop_time = std::chrono::duration_cast<std::chrono::milliseconds>
                    (std::chrono::high_resolution_clock::now() - m_times.back()).count();

        // if (std::chrono::duration_cast<std::chrono::milliseconds>
        //             (std::chrono::high_resolution_clock::now() - m_times.back()).count() > 3000)
        //     throw std::runtime_error("took too long, aborting");

        const std::string leading = leading_space ? "   " : "";

        if (m_times.size() > 1)
        {
            std::cout << leading << "execution of " << msg << " since last took " <<
                std::chrono::duration_cast<std::chrono::milliseconds>
                    (std::chrono::high_resolution_clock::now() - m_times.back()).count()
                << " ms, time since start " <<
                std::chrono::duration_cast<std::chrono::milliseconds>
                    (std::chrono::high_resolution_clock::now() - m_times.front()).count() << " ms\n";
        }
        else
        {

            std::cout << leading << "execution of " << msg << " since last took " << stop_time << " ms\n";
        }

        m_times.push_back(std::chrono::high_resolution_clock::now());
    }

    uint32_t lap_ms() const
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>
                (std::chrono::high_resolution_clock::now() - m_times.back()).count();
    }

    std::string lap(const std::string& msg = "task")
    {
        return "execution of " + msg + " took "  + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>
                (std::chrono::high_resolution_clock::now() - m_times.back()).count()) + " ms, total " + 
                std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>
                (std::chrono::high_resolution_clock::now() - m_times.front()).count()) + " ms\n";

        m_times.push_back(std::chrono::high_resolution_clock::now());
    }

    void summary() const
    {
        std::cout << m_times.size() << " laps taken\n";
        for (int i = 1; i < m_times.size(); i++)
        {
            std::cout << "lap " << i << " took: " << 
                std::chrono::duration_cast<std::chrono::milliseconds>(m_times[i] - m_times[i - 1]).count() << " ms\n";
        }

        std::cout << "to a total time of " << std::chrono::duration_cast<std::chrono::milliseconds>(m_times.back() - m_times.front()).count() << " ms\n";
    }

private:
    std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> m_times;

};

}

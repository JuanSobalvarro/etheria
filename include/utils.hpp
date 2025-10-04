#pragma once

#include <random>

namespace eth::utils
{

static std::mt19937& global_rng() 
{
    static thread_local std::mt19937 gen(std::random_device{}());
    return gen;
}

} // namespace eth::utils
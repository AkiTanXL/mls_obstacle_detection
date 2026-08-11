#pragma once

#include "mls/obstacle_detection/types.hpp"

#include <filesystem>
#include <string>

namespace mls::obstacle_detection {

PipelineConfig loadConfig(const std::filesystem::path& path);
void validateConfig(const PipelineConfig& config);
std::string configToYaml(const PipelineConfig& config);

}  // namespace mls::obstacle_detection

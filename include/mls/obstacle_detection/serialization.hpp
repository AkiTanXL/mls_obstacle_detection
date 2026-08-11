#pragma once

#include "mls/obstacle_detection/types.hpp"

#include <filesystem>
#include <string>

namespace mls::obstacle_detection {

struct OutputPaths { std::filesystem::path pcd; std::filesystem::path json; };

void prepareOutputDirectory(const std::filesystem::path& output, bool overwrite);
OutputPaths outputPathsFor(const std::filesystem::path& output, const Frame& frame);
void writeFrameResultAtomic(const OutputPaths& paths, const Frame& frame,
                            DetectionResult& result, bool binary_compressed);
void writeTextAtomic(const std::filesystem::path& path, const std::string& text);
std::string resultToJson(const Frame& frame, const DetectionResult& result);

}  // namespace mls::obstacle_detection

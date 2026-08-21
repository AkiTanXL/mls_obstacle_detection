#pragma once

#include "mls/obstacle_detection/types.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace mls::obstacle_detection {

struct OutputPaths { std::filesystem::path pcd; std::filesystem::path json; };

struct FrameSaveCompletion {
  std::filesystem::path source_path;
  PointCounts counts;
  std::size_t obstacle_count{0};
  std::size_t filtered_cluster_count{0};
  double algorithm_ms{0.0};
  double save_ms{0.0};
};

class AsyncFrameWriter {
 public:
  explicit AsyncFrameWriter(bool binary_compressed, std::size_t capacity = 2);
  ~AsyncFrameWriter();
  AsyncFrameWriter(const AsyncFrameWriter&) = delete;
  AsyncFrameWriter& operator=(const AsyncFrameWriter&) = delete;

  void enqueue(OutputPaths paths, DetectionResult result);
  std::vector<FrameSaveCompletion> takeCompletions();
  void rethrowIfFailed() const;
  void finish();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

void prepareOutputDirectory(const std::filesystem::path& output, bool overwrite);
OutputPaths outputPathsFor(const std::filesystem::path& output, const Frame& frame);
void writeFrameResultAtomic(const OutputPaths& paths, DetectionResult& result,
                            bool binary_compressed);
void writeTextAtomic(const std::filesystem::path& path, const std::string& text);
std::string resultToJson(const DetectionResult& result);

}  // namespace mls::obstacle_detection

#pragma once

#include "mls/obstacle_detection/types.hpp"

namespace mls::obstacle_detection {

class ObstacleDetectionPipeline {
 public:
  explicit ObstacleDetectionPipeline(PipelineConfig config);
  [[nodiscard]] DetectionResult process(const Frame& frame) const;

 private:
  PipelineConfig config_;
};

std::array<std::uint8_t, 3> distanceColor(float distance, const ColorConfig& config);

}  // namespace mls::obstacle_detection

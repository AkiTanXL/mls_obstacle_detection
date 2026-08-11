#pragma once

#include "mls/obstacle_detection/types.hpp"

#include <array>

namespace mls::obstacle_detection {

struct CameraPose {
  std::array<double, 3> position{};
  std::array<double, 3> focal_point{};
  std::array<double, 3> up{};
  double vertical_fov_rad{0.0};
  double near_clip{0.0};
  double far_clip{0.0};
};

// Fits the operational (non-ignored) cloud from behind and above the vehicle.
// The horizontal component of the viewing direction is always +X.
CameraPose computeForwardCamera(const DetectionResult& result);

}  // namespace mls::obstacle_detection

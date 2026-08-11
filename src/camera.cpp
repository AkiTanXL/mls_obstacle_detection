#include "mls/obstacle_detection/camera.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mls::obstacle_detection {
namespace {

struct Bounds {
  std::array<double, 3> min{{std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::infinity()}};
  std::array<double, 3> max{{-std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity()}};
  std::size_t count{0};
};

Bounds cloudBounds(const DetectionResult& result, bool operational_only) {
  Bounds bounds;
  for (const auto& point : result.labeled_cloud->points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;
    if (operational_only && point.semantic_label == static_cast<std::uint32_t>(SemanticLabel::ignored)) continue;
    const std::array<double, 3> xyz{{point.x, point.y, point.z}};
    for (std::size_t axis = 0; axis < 3; ++axis) {
      bounds.min[axis] = std::min(bounds.min[axis], xyz[axis]);
      bounds.max[axis] = std::max(bounds.max[axis], xyz[axis]);
    }
    ++bounds.count;
  }
  return bounds;
}

}  // namespace

CameraPose computeForwardCamera(const DetectionResult& result) {
  Bounds bounds = cloudBounds(result, true);
  if (bounds.count == 0) bounds = cloudBounds(result, false);
  if (bounds.count == 0) {
    bounds.min = {{-10.0, -8.0, -3.0}};
    bounds.max = {{50.0, 8.0, 3.0}};
  }

  CameraPose camera;
  std::array<double, 3> half_extent{};
  for (std::size_t axis = 0; axis < 3; ++axis) {
    camera.focal_point[axis] = 0.5 * (bounds.min[axis] + bounds.max[axis]);
    half_extent[axis] = 0.5 * (bounds.max[axis] - bounds.min[axis]);
  }
  const double radius = std::max(1.0, std::sqrt(
      half_extent[0] * half_extent[0] + half_extent[1] * half_extent[1] +
      half_extent[2] * half_extent[2]));
  constexpr double pi = 3.14159265358979323846;
  constexpr double pitch = 25.0 * pi / 180.0;
  camera.vertical_fov_rad = 50.0 * pi / 180.0;
  const std::array<double, 3> forward{{std::cos(pitch), 0.0, -std::sin(pitch)}};
  camera.up = {{std::sin(pitch), 0.0, std::cos(pitch)}};

  // Fit a bounding sphere regardless of the window aspect ratio, with margin for
  // point glyphs and obstacle boxes.
  const double distance = 1.2 * radius / std::sin(0.5 * camera.vertical_fov_rad);
  for (std::size_t axis = 0; axis < 3; ++axis) {
    camera.position[axis] = camera.focal_point[axis] - forward[axis] * distance;
  }
  camera.near_clip = std::max(0.05, distance - 1.5 * radius);
  camera.far_clip = distance + 1.5 * radius;
  return camera;
}

}  // namespace mls::obstacle_detection

#include "mls/obstacle_detection/camera.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace od = mls::obstacle_detection;

namespace {

LabeledPoint point(float x, float y, float z, od::SemanticLabel label) {
  LabeledPoint result;
  result.x = x;
  result.y = y;
  result.z = z;
  result.semantic_label = static_cast<std::uint32_t>(label);
  return result;
}

double cameraDistance(const od::CameraPose& camera) {
  double squared = 0.0;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const double delta = camera.focal_point[axis] - camera.position[axis];
    squared += delta * delta;
  }
  return std::sqrt(squared);
}

}  // namespace

TEST(Camera, FitsOperationalCloudFromBehindAlongVehicleForwardAxis) {
  od::DetectionResult result;
  result.labeled_cloud->push_back(point(0.F, -5.F, -1.F, od::SemanticLabel::ground));
  result.labeled_cloud->push_back(point(50.F, 5.F, 3.F, od::SemanticLabel::obstacle));
  // Ignored outliers must not make the useful road cloud appear tiny.
  result.labeled_cloud->push_back(point(10000.F, 10000.F, 10000.F, od::SemanticLabel::ignored));

  const auto camera = od::computeForwardCamera(result);
  EXPECT_DOUBLE_EQ(camera.focal_point[0], 25.0);
  EXPECT_DOUBLE_EQ(camera.focal_point[1], 0.0);
  EXPECT_DOUBLE_EQ(camera.focal_point[2], 1.0);
  EXPECT_LT(camera.position[0], camera.focal_point[0]);
  EXPECT_DOUBLE_EQ(camera.position[1], camera.focal_point[1]);
  EXPECT_GT(camera.position[2], camera.focal_point[2]);

  const double distance = cameraDistance(camera);
  const double radius = std::sqrt(25.0 * 25.0 + 5.0 * 5.0 + 2.0 * 2.0);
  EXPECT_LT(camera.near_clip, distance - radius);
  EXPECT_GT(camera.far_clip, distance + radius);
  EXPECT_GT(camera.vertical_fov_rad, 0.0);
}

TEST(Camera, FallsBackToFiniteIgnoredPointsAndScalesWithCloudSize) {
  od::DetectionResult small;
  small.labeled_cloud->push_back(point(0.F, 0.F, 0.F, od::SemanticLabel::ignored));
  small.labeled_cloud->push_back(point(2.F, 0.F, 0.F, od::SemanticLabel::ignored));
  od::DetectionResult large;
  large.labeled_cloud->push_back(point(-50.F, -10.F, -2.F, od::SemanticLabel::ground));
  large.labeled_cloud->push_back(point(50.F, 10.F, 2.F, od::SemanticLabel::ground));
  EXPECT_GT(cameraDistance(od::computeForwardCamera(large)),
            cameraDistance(od::computeForwardCamera(small)));
}

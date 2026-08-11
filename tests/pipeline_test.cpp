#include "mls/obstacle_detection/pipeline.hpp"

#include <gtest/gtest.h>

#include <limits>

namespace od = mls::obstacle_detection;

namespace {

od::PipelineConfig testConfig() {
  od::PipelineConfig config;
  config.roi.box = {{-20.F, -20.F, -5.F}, {30.F, 20.F, 5.F}};
  config.ego.box = {{-10.F, -10.F, -4.F}, {-9.F, -9.F, -3.F}};
  config.voxel.leaf_size = 0.1F;
  config.ground.distance_threshold = 0.08F;
  config.ground.min_inlier_ratio = 0.1;
  config.cluster.tolerance = 0.35F;
  config.cluster.min_points = 2;
  config.cluster.max_points = 20;
  return config;
}

od::Frame scene() {
  od::Frame frame;
  frame.id = "scene";
  frame.cloud->is_dense = false;
  for (int x = 0; x < 6; ++x) {
    for (int y = -3; y <= 3; ++y) {
      pcl::PointXYZI p;
      p.x = x * 0.7F; p.y = y * 0.7F; p.z = 0.03F * p.x; p.intensity = 1.F;
      frame.cloud->push_back(p);
    }
  }
  for (float x : {3.0F, 3.12F, 3.24F}) {
    pcl::PointXYZI p; p.x = x; p.y = 2.F; p.z = 1.F; p.intensity = 10.F; frame.cloud->push_back(p);
  }
  for (float x : {8.0F, 8.12F, 8.24F}) {
    pcl::PointXYZI p; p.x = x; p.y = -2.F; p.z = 1.2F; p.intensity = 20.F; frame.cloud->push_back(p);
  }
  pcl::PointXYZI outside; outside.x = 100.F; outside.y = 0.F; outside.z = 0.F; frame.cloud->push_back(outside);
  pcl::PointXYZI invalid; invalid.x = std::numeric_limits<float>::quiet_NaN(); invalid.y = 0.F; invalid.z = 0.F;
  frame.cloud->push_back(invalid);
  return frame;
}

}  // namespace

TEST(Pipeline, SegmentsTiltedGroundClustersAndLabelsOriginalResolution) {
  const auto frame = scene();
  od::ObstacleDetectionPipeline pipeline(testConfig());
  const auto result = pipeline.process(frame);
  EXPECT_EQ(result.labeled_cloud->size(), frame.cloud->size());
  ASSERT_EQ(result.obstacles.size(), 2U);
  EXPECT_EQ(result.obstacles[0].instance_id, 1U);
  EXPECT_EQ(result.obstacles[1].instance_id, 2U);
  EXPECT_LT(result.obstacles[0].sensor_distance, result.obstacles[1].sensor_distance);
  EXPECT_EQ(result.obstacles[0].point_count, 3U);
  EXPECT_FLOAT_EQ(result.obstacles[0].aabb.min[0], 3.F);
  EXPECT_FLOAT_EQ(result.obstacles[0].aabb.max[0], 3.24F);
  EXPECT_EQ(result.labeled_cloud->points[result.labeled_cloud->size() - 2].semantic_label,
            static_cast<std::uint32_t>(od::SemanticLabel::ignored));
  EXPECT_EQ(result.labeled_cloud->back().semantic_label,
            static_cast<std::uint32_t>(od::SemanticLabel::ignored));
  for (const auto& point : result.labeled_cloud->points) {
    const bool obstacle = point.semantic_label == static_cast<std::uint32_t>(od::SemanticLabel::obstacle);
    EXPECT_EQ(point.instance_id > 0, obstacle);
  }
}

TEST(Pipeline, VoxelManyToOneWritesBackEveryRawPoint) {
  auto config = testConfig();
  config.voxel.leaf_size = 0.5F;
  config.cluster.min_points = 1;
  od::Frame frame;
  frame.id = "voxel";
  for (int x = 0; x < 6; ++x) for (int y = 0; y < 6; ++y) {
    pcl::PointXYZI p; p.x = x; p.y = y; p.z = 0.F; frame.cloud->push_back(p);
  }
  for (float delta : {0.F, 0.05F, 0.1F, 0.15F}) {
    pcl::PointXYZI p; p.x = 8.F + delta; p.y = 1.F; p.z = 1.F; frame.cloud->push_back(p);
  }
  const auto result = od::ObstacleDetectionPipeline(config).process(frame);
  ASSERT_EQ(result.obstacles.size(), 1U);
  EXPECT_EQ(result.obstacles[0].point_count, 4U);
  for (std::size_t i = frame.cloud->size() - 4; i < frame.cloud->size(); ++i) {
    EXPECT_EQ(result.labeled_cloud->points[i].instance_id, 1U);
  }
}

TEST(Color, HasMagmaEndpointsAndClamps) {
  od::ColorConfig config;
  config.palette = "magma";
  EXPECT_EQ(od::distanceColor(-1.F, config), (std::array<std::uint8_t, 3>{0, 0, 4}));
  EXPECT_EQ(od::distanceColor(50.F, config), (std::array<std::uint8_t, 3>{252, 253, 191}));
  EXPECT_EQ(od::distanceColor(500.F, config), (std::array<std::uint8_t, 3>{252, 253, 191}));
}

TEST(Color, RedGreenRunsFromNearRedThroughYellowToFarGreen) {
  od::ColorConfig config;
  config.palette = "red_green";
  EXPECT_EQ(od::distanceColor(-1.F, config), (std::array<std::uint8_t, 3>{255, 0, 0}));
  EXPECT_EQ(od::distanceColor(25.F, config), (std::array<std::uint8_t, 3>{128, 128, 0}));
  EXPECT_EQ(od::distanceColor(50.F, config), (std::array<std::uint8_t, 3>{0, 255, 0}));
  EXPECT_EQ(od::distanceColor(500.F, config), (std::array<std::uint8_t, 3>{0, 255, 0}));
}

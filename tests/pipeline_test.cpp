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

od::Frame elongatedInteriorScene() {
  od::Frame frame;
  frame.id = "elongated_interior";
  for (int x = 0; x <= 30; ++x) {
    for (int y = -4; y <= 4; ++y) {
      pcl::PointXYZI point;
      point.x = static_cast<float>(x);
      point.y = static_cast<float>(y);
      point.z = 0.F;
      frame.cloud->push_back(point);
    }
  }
  for (int step = 0; step <= 96; ++step) {
    const float distance = 0.25F * static_cast<float>(step);
    pcl::PointXYZI point;
    point.x = 5.F + distance;
    point.y = 6.F;
    point.z = 1.F;
    frame.cloud->push_back(point);
  }
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

TEST(Pipeline, FiltersEveryClusterAtHorizontalBoundaryAndRenumbersObstacles) {
  auto config = testConfig();
  config.roi.box.max[1] = 2.5F;
  config.cluster_filter.roi_boundary_margin = 0.5F;
  const auto result = od::ObstacleDetectionPipeline(config).process(scene());

  ASSERT_EQ(result.obstacles.size(), 1U);
  EXPECT_FLOAT_EQ(result.obstacles[0].aabb.min[0], 8.F);
  EXPECT_EQ(result.obstacles[0].instance_id, 1U);
  ASSERT_EQ(result.filtered_clusters.size(), 1U);
  const auto& filtered = result.filtered_clusters[0];
  EXPECT_EQ(filtered.point_count, 3U);
  EXPECT_EQ(filtered.boundary_faces, (std::vector<std::string>{"y_max"}));
  EXPECT_EQ(result.counts.filtered_cluster_points, 3U);
  for (std::size_t index = 42; index < 45; ++index) {
    EXPECT_EQ(result.labeled_cloud->points[index].semantic_label,
              static_cast<std::uint32_t>(od::SemanticLabel::non_ground_unclustered));
    EXPECT_EQ(result.labeled_cloud->points[index].instance_id, 0U);
    EXPECT_EQ(result.labeled_cloud->points[index].rgba, od::packRgba(config.color.unclustered));
  }
}

TEST(Pipeline, KeepsClusterOutsideBoundaryMarginAndWhenFilterDisabled) {
  auto config = testConfig();
  config.roi.box.max[1] = 2.51F;
  config.cluster_filter.roi_boundary_margin = 0.5F;
  auto result = od::ObstacleDetectionPipeline(config).process(scene());
  EXPECT_EQ(result.obstacles.size(), 2U);
  EXPECT_TRUE(result.filtered_clusters.empty());

  config.roi.box.max[1] = 2.1F;
  config.cluster_filter.enabled = false;
  result = od::ObstacleDetectionPipeline(config).process(scene());
  EXPECT_EQ(result.obstacles.size(), 2U);
  EXPECT_TRUE(result.filtered_clusters.empty());
}

TEST(Pipeline, RecordsEveryTouchedHorizontalFaceAndIgnoresVerticalBoundary) {
  auto corner_config = testConfig();
  corner_config.roi.box.max[0] = 3.5F;
  corner_config.roi.box.max[1] = 2.5F;
  corner_config.cluster_filter.roi_boundary_margin = 0.5F;
  auto result = od::ObstacleDetectionPipeline(corner_config).process(scene());
  ASSERT_EQ(result.filtered_clusters.size(), 1U);
  EXPECT_EQ(result.filtered_clusters[0].boundary_faces,
            (std::vector<std::string>{"x_max", "y_max"}));

  auto vertical_config = testConfig();
  vertical_config.roi.box.max[2] = 1.2F;
  result = od::ObstacleDetectionPipeline(vertical_config).process(scene());
  EXPECT_EQ(result.obstacles.size(), 2U);
  EXPECT_TRUE(result.filtered_clusters.empty());
}

TEST(Pipeline, KeepsLargeInteriorClusterWithoutGeometryThresholds) {
  auto config = testConfig();
  config.roi.box = {{-5.F, -10.F, -2.F}, {35.F, 10.F, 1.1F}};
  config.ego.box = {{-5.F, -10.F, -2.F}, {-4.F, -9.F, -1.F}};
  config.voxel.leaf_size = 0.05F;
  config.cluster.tolerance = 0.3F;
  config.cluster.max_points = 150;
  const auto result = od::ObstacleDetectionPipeline(config).process(elongatedInteriorScene());

  ASSERT_EQ(result.obstacles.size(), 1U);
  EXPECT_EQ(result.obstacles[0].point_count, 97U);
  EXPECT_FLOAT_EQ(result.obstacles[0].aabb.max[0] - result.obstacles[0].aabb.min[0], 24.F);
  EXPECT_TRUE(result.filtered_clusters.empty());
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

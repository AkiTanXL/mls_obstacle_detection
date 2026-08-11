#include "mls/obstacle_detection/frame_source.hpp"
#include "mls/obstacle_detection/pipeline.hpp"

#include <gtest/gtest.h>

#include <cstdlib>

namespace od = mls::obstacle_detection;

TEST(DatasetIntegration, ExternalHighwayDatasetIfConfigured) {
  const char* variable = std::getenv("MLS_HIGHWAY_DATASET");
  if (variable == nullptr) GTEST_SKIP() << "MLS_HIGHWAY_DATASET is not set";
  const std::filesystem::path root(variable);
  const std::vector<std::pair<std::filesystem::path, std::size_t>> inputs{
      {root / "simpleHighway.pcd", 1}, {root / "data_1", 22}, {root / "data_2", 154}};
  od::ObstacleDetectionPipeline pipeline(od::PipelineConfig{});
  for (const auto& [path, expected] : inputs) {
    ASSERT_TRUE(std::filesystem::exists(path)) << path;
    od::PcdFrameSource source(path);
    EXPECT_EQ(source.size(), expected) << path;
    od::Frame frame;
    while (source.next(frame)) {
      const auto result = pipeline.process(frame);
      const auto repeated = pipeline.process(frame);
      ASSERT_EQ(result.labeled_cloud->size(), frame.cloud->size());
      ASSERT_EQ(result.obstacles.size(), repeated.obstacles.size());
      for (const auto& point : result.labeled_cloud->points) {
        ASSERT_LE(point.semantic_label, 3U);
        ASSERT_EQ(point.instance_id > 0, point.semantic_label == 3U);
      }
      for (std::size_t i = 0; i < result.labeled_cloud->size(); ++i) {
        EXPECT_EQ(result.labeled_cloud->points[i].semantic_label, repeated.labeled_cloud->points[i].semantic_label);
        EXPECT_EQ(result.labeled_cloud->points[i].instance_id, repeated.labeled_cloud->points[i].instance_id);
        EXPECT_EQ(result.labeled_cloud->points[i].rgba, repeated.labeled_cloud->points[i].rgba);
      }
      for (std::size_t i = 0; i < result.obstacles.size(); ++i) {
        EXPECT_EQ(result.obstacles[i].instance_id, repeated.obstacles[i].instance_id);
        EXPECT_EQ(result.obstacles[i].aabb.min, repeated.obstacles[i].aabb.min);
        EXPECT_EQ(result.obstacles[i].aabb.max, repeated.obstacles[i].aabb.max);
      }
    }
  }
}

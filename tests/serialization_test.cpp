#include "mls/obstacle_detection/serialization.hpp"
#include "test_support.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <pcl/PCLPointCloud2.h>
#include <pcl/io/pcd_io.h>

#include <fstream>

namespace od = mls::obstacle_detection;

TEST(Serialization, WritesExpectedFieldsOrderViewpointAndJsonSchema) {
  TemporaryDirectory temporary;
  od::Frame frame;
  frame.id = "frame_1";
  frame.source_path = "input/frame_1.pcd";
  frame.sensor_origin = Eigen::Vector4f(1.F, 2.F, 3.F, 0.F);
  frame.sensor_orientation = Eigen::Quaternionf(Eigen::AngleAxisf(0.3F, Eigen::Vector3f::UnitZ()));
  frame.cloud->width = 2; frame.cloud->height = 1; frame.cloud->resize(2);
  frame.cloud->points[0].x = 4.F; frame.cloud->points[0].intensity = 7.F;
  frame.cloud->points[1].x = 5.F; frame.cloud->points[1].intensity = 8.F;
  od::DetectionResult result;
  result.frame_id = frame.id; result.source_path = frame.source_path; result.counts.input = 2;
  result.labeled_cloud->width = 2; result.labeled_cloud->height = 1; result.labeled_cloud->resize(2);
  result.labeled_cloud->sensor_origin_ = frame.sensor_origin;
  result.labeled_cloud->sensor_orientation_ = frame.sensor_orientation;
  for (std::size_t i = 0; i < 2; ++i) {
    result.labeled_cloud->points[i].x = frame.cloud->points[i].x;
    result.labeled_cloud->points[i].intensity = frame.cloud->points[i].intensity;
    result.labeled_cloud->points[i].semantic_label = i == 0 ? 1U : 3U;
    result.labeled_cloud->points[i].instance_id = i == 0 ? 0U : 1U;
  }
  od::Obstacle obstacle;
  obstacle.instance_id = 1;
  obstacle.point_count = 1;
  obstacle.aabb = {{5.F, 0.F, 0.F}, {5.F, 0.F, 0.F}};
  obstacle.centroid = {{5.F, 0.F, 0.F}};
  obstacle.sensor_distance = 5.F;
  obstacle.color = {{10, 20, 30}};
  result.obstacles.push_back(obstacle);
  const auto paths = od::outputPathsFor(temporary.path(), frame);
  od::writeFrameResultAtomic(paths, frame, result, true);
  pcl::PCLPointCloud2 blob;
  Eigen::Vector4f origin;
  Eigen::Quaternionf orientation;
  ASSERT_EQ(pcl::io::loadPCDFile(paths.pcd.string(), blob, origin, orientation), 0);
  ASSERT_EQ(blob.width * blob.height, 2U);
  const std::vector<std::string> expected{"x", "y", "z", "intensity", "rgba", "semantic_label", "instance_id"};
  ASSERT_EQ(blob.fields.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) EXPECT_EQ(blob.fields[i].name, expected[i]);
  EXPECT_TRUE(origin.isApprox(frame.sensor_origin));
  EXPECT_TRUE(orientation.isApprox(frame.sensor_orientation));
  pcl::PointCloud<LabeledPoint> loaded;
  ASSERT_EQ(pcl::io::loadPCDFile(paths.pcd.string(), loaded), 0);
  ASSERT_EQ(loaded.size(), 2U);
  EXPECT_FLOAT_EQ(loaded.points[0].x, 4.F);
  EXPECT_FLOAT_EQ(loaded.points[1].x, 5.F);
  EXPECT_EQ(loaded.points[0].semantic_label, 1U);
  EXPECT_EQ(loaded.points[0].instance_id, 0U);
  EXPECT_EQ(loaded.points[1].semantic_label, 3U);
  EXPECT_EQ(loaded.points[1].instance_id, 1U);
  std::ifstream input(paths.json);
  const auto json = nlohmann::json::parse(input);
  EXPECT_EQ(json.at("schema_version"), 1);
  EXPECT_EQ(json.at("point_counts").at("input"), 2);
  EXPECT_EQ(json.at("obstacle_count"), 1);
  EXPECT_EQ(json.at("obstacles").at(0).at("point_count"), 1);
  const auto& saved_box = json.at("obstacles").at(0).at("aabb");
  EXPECT_LE(saved_box.at("min").at(0).get<float>(), loaded.points[1].x);
  EXPECT_GE(saved_box.at("max").at(0).get<float>(), loaded.points[1].x);
  EXPECT_TRUE(json.contains("timings_ms"));
}

TEST(Serialization, AtomicWriteFailureDoesNotLeaveTargets) {
  TemporaryDirectory temporary;
  od::Frame frame;
  frame.id = "missing_parent";
  od::DetectionResult result;
  result.frame_id = frame.id;
  const auto missing = temporary.path() / "does-not-exist";
  const auto paths = od::outputPathsFor(missing, frame);
  EXPECT_THROW(od::writeFrameResultAtomic(paths, frame, result, true), std::runtime_error);
  EXPECT_FALSE(std::filesystem::exists(paths.pcd));
  EXPECT_FALSE(std::filesystem::exists(paths.json));
}

TEST(Serialization, RejectsNonEmptyOutputUnlessOverwrite) {
  TemporaryDirectory temporary;
  { std::ofstream output(temporary.path() / "existing"); output << "x"; }
  EXPECT_THROW(od::prepareOutputDirectory(temporary.path(), false), std::runtime_error);
  EXPECT_NO_THROW(od::prepareOutputDirectory(temporary.path(), true));
}

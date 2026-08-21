#include "mls/obstacle_detection/serialization.hpp"
#include "test_support.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <pcl/PCLPointCloud2.h>
#include <pcl/io/pcd_io.h>

#include <fstream>
#include <string>
#include <utility>

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
  od::FilteredCluster filtered;
  filtered.point_count = 12;
  filtered.aabb = {{-2.F, 7.8F, 0.F}, {9.F, 8.F, 3.F}};
  filtered.boundary_faces = {"y_max"};
  result.filtered_clusters.push_back(filtered);
  result.counts.filtered_cluster_points = 12;
  const auto paths = od::outputPathsFor(temporary.path(), frame);
  od::writeFrameResultAtomic(paths, result, true);
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
  EXPECT_EQ(json.at("schema_version"), 3);
  EXPECT_EQ(json.at("point_counts").at("input"), 2);
  EXPECT_EQ(json.at("point_counts").at("filtered_cluster_raw"), 12);
  EXPECT_EQ(json.at("obstacle_count"), 1);
  EXPECT_EQ(json.at("obstacles").at(0).at("point_count"), 1);
  const auto& saved_box = json.at("obstacles").at(0).at("aabb");
  EXPECT_LE(saved_box.at("min").at(0).get<float>(), loaded.points[1].x);
  EXPECT_GE(saved_box.at("max").at(0).get<float>(), loaded.points[1].x);
  EXPECT_TRUE(json.contains("timings_ms"));
  EXPECT_EQ(json.at("filtered_cluster_count"), 1);
  const auto& filtered_json = json.at("filtered_clusters").at(0);
  EXPECT_EQ(filtered_json.at("point_count"), 12);
  EXPECT_EQ(filtered_json.at("boundary_faces").at(0), "y_max");
  EXPECT_EQ(filtered_json.size(), 3U);
  EXPECT_FALSE(filtered_json.contains("horizontal_extents"));
  EXPECT_FALSE(filtered_json.contains("reasons"));
}

TEST(Serialization, AtomicWriteFailureDoesNotLeaveTargets) {
  TemporaryDirectory temporary;
  od::Frame frame;
  frame.id = "missing_parent";
  od::DetectionResult result;
  result.frame_id = frame.id;
  const auto missing = temporary.path() / "does-not-exist";
  const auto paths = od::outputPathsFor(missing, frame);
  EXPECT_THROW(od::writeFrameResultAtomic(paths, result, true), std::runtime_error);
  EXPECT_FALSE(std::filesystem::exists(paths.pcd));
  EXPECT_FALSE(std::filesystem::exists(paths.json));
}

TEST(Serialization, RejectsNonEmptyOutputUnlessOverwrite) {
  TemporaryDirectory temporary;
  { std::ofstream output(temporary.path() / "existing"); output << "x"; }
  EXPECT_THROW(od::prepareOutputDirectory(temporary.path(), false), std::runtime_error);
  EXPECT_NO_THROW(od::prepareOutputDirectory(temporary.path(), true));
}

TEST(Serialization, AsyncWriterDrainsFramesInSubmissionOrder) {
  TemporaryDirectory temporary;
  od::AsyncFrameWriter writer(true, 2);
  for (int index = 0; index < 3; ++index) {
    od::DetectionResult result;
    result.frame_id = "async_" + std::to_string(index);
    result.source_path = std::filesystem::path("input") / (result.frame_id + ".pcd");
    result.counts.input = 1;
    result.timings.algorithm_ms = static_cast<double>(index + 1);
    result.labeled_cloud->width = 1;
    result.labeled_cloud->height = 1;
    result.labeled_cloud->resize(1);
    result.labeled_cloud->points[0].x = static_cast<float>(index);
    const od::OutputPaths paths{temporary.path() / (result.frame_id + ".pcd"),
                                temporary.path() / (result.frame_id + ".json")};
    writer.enqueue(paths, std::move(result));
  }
  writer.finish();

  const auto completions = writer.takeCompletions();
  ASSERT_EQ(completions.size(), 3U);
  for (std::size_t index = 0; index < completions.size(); ++index) {
    const std::string id = "async_" + std::to_string(index);
    EXPECT_EQ(completions[index].source_path.filename(), id + ".pcd");
    EXPECT_DOUBLE_EQ(completions[index].algorithm_ms, static_cast<double>(index + 1));
    EXPECT_GE(completions[index].save_ms, 0.0);
    EXPECT_TRUE(std::filesystem::exists(temporary.path() / (id + ".pcd")));
    EXPECT_TRUE(std::filesystem::exists(temporary.path() / (id + ".json")));
  }
  EXPECT_TRUE(writer.takeCompletions().empty());
}

TEST(Serialization, AsyncWriterPropagatesSaveFailure) {
  TemporaryDirectory temporary;
  od::AsyncFrameWriter writer(true);
  od::DetectionResult result;
  result.frame_id = "failure";
  result.source_path = "input/failure.pcd";
  const auto missing = temporary.path() / "missing";
  writer.enqueue({missing / "failure.pcd", missing / "failure.json"}, std::move(result));
  EXPECT_THROW(writer.finish(), std::runtime_error);
  EXPECT_THROW(writer.rethrowIfFailed(), std::runtime_error);
  EXPECT_FALSE(std::filesystem::exists(missing / "failure.pcd"));
  EXPECT_FALSE(std::filesystem::exists(missing / "failure.json"));
}

TEST(Serialization, AsyncWriterRejectsZeroCapacity) {
  EXPECT_THROW(od::AsyncFrameWriter(true, 0), std::invalid_argument);
}

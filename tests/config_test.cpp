#include "mls/obstacle_detection/config.hpp"
#include "test_support.hpp"

#include <gtest/gtest.h>

#include <array>
#include <fstream>
#include <string>
#include <vector>

namespace od = mls::obstacle_detection;

TEST(Config, DefaultsAndRoundTrip) {
  TemporaryDirectory temporary;
  const auto path = temporary.path() / "minimal.yaml";
  { std::ofstream output(path); output << "{}\n"; }
  const auto config = od::loadConfig(path);
  EXPECT_EQ(config.roi.box.min, (std::array<float, 3>{-15.F, -15.F, -3.F}));
  EXPECT_EQ(config.roi.box.max, (std::array<float, 3>{70.F, 15.F, 4.F}));
  EXPECT_FLOAT_EQ(config.voxel.leaf_size, 0.2F);
  EXPECT_EQ(config.cluster.min_points, 10);
  EXPECT_TRUE(config.cluster_filter.enabled);
  EXPECT_FLOAT_EQ(config.cluster_filter.roi_boundary_margin, 0.5F);
  EXPECT_DOUBLE_EQ(config.playback.fps, 10.0);
  const std::string resolved = od::configToYaml(config);
  EXPECT_NE(resolved.find("perpendicular_plane"), std::string::npos);
  EXPECT_NE(resolved.find("cluster_filter"), std::string::npos);
  EXPECT_EQ(resolved.find("max_horizontal_aspect_ratio"), std::string::npos);
}

TEST(Config, ReadsClusterFilter) {
  TemporaryDirectory temporary;
  const auto path = temporary.path() / "filter.yaml";
  { std::ofstream output(path); output << R"(
cluster_filter:
  enabled: false
  roi_boundary_margin: 0.4
)"; }
  const auto config = od::loadConfig(path);
  EXPECT_FALSE(config.cluster_filter.enabled);
  EXPECT_FLOAT_EQ(config.cluster_filter.roi_boundary_margin, 0.4F);
}

TEST(Config, RejectsRemovedClusterFilterThresholds) {
  TemporaryDirectory temporary;
  const std::vector<std::string> removed{
      "max_horizontal_major_extent", "max_horizontal_minor_extent", "max_height",
      "max_horizontal_aspect_ratio", "min_major_extent_for_aspect_ratio"};
  for (const auto& key : removed) {
    const auto path = temporary.path() / (key + ".yaml");
    { std::ofstream output(path); output << "cluster_filter: {" << key << ": 1.0}\n"; }
    EXPECT_THROW(od::loadConfig(path), std::invalid_argument) << key;
  }
}

TEST(Config, RejectsInvalidBoundariesAndRanges) {
  od::PipelineConfig config;
  config.roi.box.min[0] = config.roi.box.max[0];
  EXPECT_THROW(od::validateConfig(config), std::invalid_argument);
  config = {};
  config.ground.probability = 1.0;
  EXPECT_THROW(od::validateConfig(config), std::invalid_argument);
  config = {};
  config.cluster.max_points = config.cluster.min_points - 1;
  EXPECT_THROW(od::validateConfig(config), std::invalid_argument);
  config = {};
  config.cluster_filter.roi_boundary_margin = 15.F;
  EXPECT_THROW(od::validateConfig(config), std::invalid_argument);
  config = {};
  config.roi.box.max[1] = config.roi.box.min[1] + 0.5F;
  EXPECT_THROW(od::validateConfig(config), std::invalid_argument);
  config.cluster_filter.enabled = false;
  EXPECT_NO_THROW(od::validateConfig(config));
  config = {};
  config.color.max_distance = config.color.min_distance;
  EXPECT_THROW(od::validateConfig(config), std::invalid_argument);
}

TEST(Config, RejectsUnsupportedGroundModelAndAxis) {
  TemporaryDirectory temporary;
  const auto model = temporary.path() / "model.yaml";
  { std::ofstream output(model); output << "ground: {model: plane}\n"; }
  EXPECT_THROW(od::loadConfig(model), std::invalid_argument);
  const auto axis = temporary.path() / "axis.yaml";
  { std::ofstream output(axis); output << "ground: {axis: [1, 0, 0]}\n"; }
  EXPECT_THROW(od::loadConfig(axis), std::invalid_argument);
}

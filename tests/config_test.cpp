#include "mls/obstacle_detection/config.hpp"
#include "test_support.hpp"

#include <gtest/gtest.h>

#include <fstream>

namespace od = mls::obstacle_detection;

TEST(Config, DefaultsAndRoundTrip) {
  TemporaryDirectory temporary;
  const auto path = temporary.path() / "minimal.yaml";
  { std::ofstream output(path); output << "{}\n"; }
  const auto config = od::loadConfig(path);
  EXPECT_FLOAT_EQ(config.roi.box.min[0], -10.F);
  EXPECT_FLOAT_EQ(config.voxel.leaf_size, 0.2F);
  EXPECT_EQ(config.cluster.min_points, 10);
  EXPECT_DOUBLE_EQ(config.playback.fps, 10.0);
  const std::string resolved = od::configToYaml(config);
  EXPECT_NE(resolved.find("perpendicular_plane"), std::string::npos);
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

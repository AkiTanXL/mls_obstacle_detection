#include "mls/obstacle_detection/config.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace mls::obstacle_detection {
namespace {

template <typename T>
void assignIfPresent(const YAML::Node& parent, const char* key, T& value) {
  if (const auto node = parent[key]) value = node.as<T>();
}

void readBox(const YAML::Node& node, Box3f& box, const std::string& name) {
  if (!node) return;
  const auto read = [&](const char* key, std::array<float, 3>& target) {
    const auto value = node[key];
    if (!value) return;
    if (!value.IsSequence() || value.size() != 3) {
      throw std::invalid_argument(name + "." + key + " must contain exactly 3 numbers");
    }
    for (std::size_t i = 0; i < 3; ++i) target[i] = value[i].as<float>();
  };
  read("min", box.min);
  read("max", box.max);
}

void checkFinite(float value, const std::string& name) {
  if (!std::isfinite(value)) throw std::invalid_argument(name + " must be finite");
}

void emitBox(YAML::Emitter& out, const char* name, const Box3f& box) {
  out << YAML::Key << name << YAML::Value << YAML::BeginMap;
  out << YAML::Key << "min" << YAML::Value << YAML::Flow << YAML::BeginSeq;
  for (float v : box.min) out << v;
  out << YAML::EndSeq;
  out << YAML::Key << "max" << YAML::Value << YAML::Flow << YAML::BeginSeq;
  for (float v : box.max) out << v;
  out << YAML::EndSeq << YAML::EndMap;
}

}  // namespace

PipelineConfig loadConfig(const std::filesystem::path& path) {
  PipelineConfig config;
  YAML::Node root;
  try {
    root = YAML::LoadFile(path.string());
  } catch (const YAML::Exception& error) {
    throw std::runtime_error("cannot load config '" + path.string() + "': " + error.what());
  }
  if (!root.IsMap()) throw std::invalid_argument("config root must be a mapping");

  readBox(root["roi"], config.roi.box, "roi");
  readBox(root["ego_box"], config.ego.box, "ego_box");
  if (const auto node = root["voxel"]) {
    assignIfPresent(node, "leaf_size", config.voxel.leaf_size);
    assignIfPresent(node, "min_points_per_voxel", config.voxel.min_points_per_voxel);
  }
  if (const auto node = root["ground"]) {
    if (const auto model = node["model"]; model && model.as<std::string>() != "perpendicular_plane") {
      throw std::invalid_argument("ground.model must be perpendicular_plane");
    }
    if (const auto axis = node["axis"]) {
      if (!axis.IsSequence() || axis.size() != 3 || axis[0].as<float>() != 0.F ||
          axis[1].as<float>() != 0.F || axis[2].as<float>() != 1.F) {
        throw std::invalid_argument("ground.axis must be [0, 0, 1]");
      }
    }
    assignIfPresent(node, "max_angle_deg", config.ground.max_angle_deg);
    assignIfPresent(node, "distance_threshold", config.ground.distance_threshold);
    assignIfPresent(node, "max_iterations", config.ground.max_iterations);
    assignIfPresent(node, "probability", config.ground.probability);
    assignIfPresent(node, "optimize_coefficients", config.ground.optimize_coefficients);
    assignIfPresent(node, "min_inlier_ratio", config.ground.min_inlier_ratio);
  }
  if (const auto node = root["cluster"]) {
    assignIfPresent(node, "tolerance", config.cluster.tolerance);
    assignIfPresent(node, "min_points", config.cluster.min_points);
    assignIfPresent(node, "max_points", config.cluster.max_points);
  }
  if (const auto node = root["cluster_filter"]) {
    static constexpr std::array<const char*, 5> removed_keys{{
        "max_horizontal_major_extent", "max_horizontal_minor_extent", "max_height",
        "max_horizontal_aspect_ratio", "min_major_extent_for_aspect_ratio"}};
    for (const char* key : removed_keys) {
      if (node[key]) {
        throw std::invalid_argument(
            std::string("cluster_filter.") + key +
            " is no longer supported; boundary filtering now uses only enabled and roi_boundary_margin");
      }
    }
    assignIfPresent(node, "enabled", config.cluster_filter.enabled);
    assignIfPresent(node, "roi_boundary_margin", config.cluster_filter.roi_boundary_margin);
  }
  if (const auto node = root["color"]) {
    assignIfPresent(node, "min_distance", config.color.min_distance);
    assignIfPresent(node, "max_distance", config.color.max_distance);
    assignIfPresent(node, "palette", config.color.palette);
    assignIfPresent(node, "clamp", config.color.clamp);
  }
  if (const auto node = root["save"]) {
    assignIfPresent(node, "binary_compressed", config.save.binary_compressed);
  }
  if (const auto node = root["playback"]) {
    assignIfPresent(node, "fps", config.playback.fps);
    assignIfPresent(node, "loop", config.playback.loop);
  }
  validateConfig(config);
  return config;
}

void validateConfig(const PipelineConfig& config) {
  const auto validate_box = [](const Box3f& box, const std::string& name) {
    for (std::size_t i = 0; i < 3; ++i) {
      checkFinite(box.min[i], name + ".min");
      checkFinite(box.max[i], name + ".max");
      if (box.min[i] >= box.max[i]) {
        throw std::invalid_argument(name + ".min must be less than " + name + ".max on every axis");
      }
    }
  };
  validate_box(config.roi.box, "roi");
  validate_box(config.ego.box, "ego_box");
  checkFinite(config.voxel.leaf_size, "voxel.leaf_size");
  if (config.voxel.leaf_size <= 0.F) throw std::invalid_argument("voxel.leaf_size must be positive");
  if (config.voxel.min_points_per_voxel == 0) {
    throw std::invalid_argument("voxel.min_points_per_voxel must be positive");
  }
  checkFinite(config.ground.max_angle_deg, "ground.max_angle_deg");
  if (config.ground.max_angle_deg <= 0.F || config.ground.max_angle_deg >= 90.F) {
    throw std::invalid_argument("ground.max_angle_deg must be in (0, 90)");
  }
  checkFinite(config.ground.distance_threshold, "ground.distance_threshold");
  if (config.ground.distance_threshold <= 0.F) {
    throw std::invalid_argument("ground.distance_threshold must be positive");
  }
  if (config.ground.max_iterations <= 0) throw std::invalid_argument("ground.max_iterations must be positive");
  if (!(config.ground.probability > 0.0 && config.ground.probability < 1.0)) {
    throw std::invalid_argument("ground.probability must be in (0, 1)");
  }
  if (!(config.ground.min_inlier_ratio > 0.0 && config.ground.min_inlier_ratio <= 1.0)) {
    throw std::invalid_argument("ground.min_inlier_ratio must be in (0, 1]");
  }
  checkFinite(config.cluster.tolerance, "cluster.tolerance");
  if (config.cluster.tolerance <= 0.F) throw std::invalid_argument("cluster.tolerance must be positive");
  if (config.cluster.min_points <= 0 || config.cluster.max_points < config.cluster.min_points) {
    throw std::invalid_argument("cluster point range must be positive and ordered");
  }
  const auto& filter = config.cluster_filter;
  checkFinite(filter.roi_boundary_margin, "cluster_filter.roi_boundary_margin");
  const float minimum_horizontal_span = std::min(
      config.roi.box.max[0] - config.roi.box.min[0],
      config.roi.box.max[1] - config.roi.box.min[1]);
  if (filter.roi_boundary_margin <= 0.F) {
    throw std::invalid_argument("cluster_filter.roi_boundary_margin must be positive");
  }
  if (filter.enabled && 2.F * filter.roi_boundary_margin >= minimum_horizontal_span) {
    throw std::invalid_argument(
        "cluster_filter.roi_boundary_margin must be less than half the minimum horizontal ROI span when enabled");
  }
  checkFinite(config.color.min_distance, "color.min_distance");
  checkFinite(config.color.max_distance, "color.max_distance");
  if (config.color.min_distance < 0.F || config.color.min_distance >= config.color.max_distance) {
    throw std::invalid_argument("color distance range must be non-negative and ordered");
  }
  if (config.color.palette != "magma" && config.color.palette != "red_green") {
    throw std::invalid_argument("color.palette must be magma or red_green");
  }
  if (!config.color.clamp) throw std::invalid_argument("color.clamp=false is not supported");
  if (!std::isfinite(config.playback.fps) || config.playback.fps <= 0.0) {
    throw std::invalid_argument("playback.fps must be positive");
  }
}

std::string configToYaml(const PipelineConfig& c) {
  YAML::Emitter out;
  out << YAML::BeginMap;
  emitBox(out, "roi", c.roi.box);
  emitBox(out, "ego_box", c.ego.box);
  out << YAML::Key << "voxel" << YAML::Value << YAML::BeginMap
      << YAML::Key << "leaf_size" << YAML::Value << c.voxel.leaf_size
      << YAML::Key << "min_points_per_voxel" << YAML::Value << c.voxel.min_points_per_voxel << YAML::EndMap;
  out << YAML::Key << "ground" << YAML::Value << YAML::BeginMap
      << YAML::Key << "model" << YAML::Value << "perpendicular_plane"
      << YAML::Key << "axis" << YAML::Value << YAML::Flow << YAML::BeginSeq << 0 << 0 << 1 << YAML::EndSeq
      << YAML::Key << "max_angle_deg" << YAML::Value << c.ground.max_angle_deg
      << YAML::Key << "distance_threshold" << YAML::Value << c.ground.distance_threshold
      << YAML::Key << "max_iterations" << YAML::Value << c.ground.max_iterations
      << YAML::Key << "probability" << YAML::Value << c.ground.probability
      << YAML::Key << "optimize_coefficients" << YAML::Value << c.ground.optimize_coefficients
      << YAML::Key << "min_inlier_ratio" << YAML::Value << c.ground.min_inlier_ratio << YAML::EndMap;
  out << YAML::Key << "cluster" << YAML::Value << YAML::BeginMap
      << YAML::Key << "tolerance" << YAML::Value << c.cluster.tolerance
      << YAML::Key << "min_points" << YAML::Value << c.cluster.min_points
      << YAML::Key << "max_points" << YAML::Value << c.cluster.max_points << YAML::EndMap;
  out << YAML::Key << "cluster_filter" << YAML::Value << YAML::BeginMap
      << YAML::Key << "enabled" << YAML::Value << c.cluster_filter.enabled
      << YAML::Key << "roi_boundary_margin" << YAML::Value
      << c.cluster_filter.roi_boundary_margin << YAML::EndMap;
  out << YAML::Key << "color" << YAML::Value << YAML::BeginMap
      << YAML::Key << "palette" << YAML::Value << c.color.palette
      << YAML::Key << "min_distance" << YAML::Value << c.color.min_distance
      << YAML::Key << "max_distance" << YAML::Value << c.color.max_distance
      << YAML::Key << "clamp" << YAML::Value << c.color.clamp << YAML::EndMap;
  out << YAML::Key << "save" << YAML::Value << YAML::BeginMap
      << YAML::Key << "binary_compressed" << YAML::Value << c.save.binary_compressed << YAML::EndMap;
  out << YAML::Key << "playback" << YAML::Value << YAML::BeginMap
      << YAML::Key << "fps" << YAML::Value << c.playback.fps
      << YAML::Key << "loop" << YAML::Value << c.playback.loop << YAML::EndMap;
  out << YAML::EndMap;
  if (!out.good()) throw std::runtime_error("failed to serialize resolved config");
  return std::string(out.c_str()) + "\n";
}

}  // namespace mls::obstacle_detection

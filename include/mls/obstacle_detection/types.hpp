#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/register_point_struct.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

struct EIGEN_ALIGN16 LabeledPoint {
  PCL_ADD_POINT4D;
  float intensity{0.0F};
  std::uint32_t rgba{0U};
  std::uint32_t semantic_label{0U};
  std::uint32_t instance_id{0U};
  PCL_MAKE_ALIGNED_OPERATOR_NEW
};

POINT_CLOUD_REGISTER_POINT_STRUCT(
    LabeledPoint,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
    (std::uint32_t, rgba, rgba)(std::uint32_t, semantic_label, semantic_label)
    (std::uint32_t, instance_id, instance_id))

namespace mls::obstacle_detection {

enum class SemanticLabel : std::uint32_t {
  ignored = 0,
  ground = 1,
  non_ground_unclustered = 2,
  obstacle = 3,
};

struct Frame {
  std::string id;
  std::filesystem::path source_path;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud{new pcl::PointCloud<pcl::PointXYZI>};
  Eigen::Vector4f sensor_origin{Eigen::Vector4f::Zero()};
  Eigen::Quaternionf sensor_orientation{Eigen::Quaternionf::Identity()};
};

struct Box3f {
  std::array<float, 3> min{};
  std::array<float, 3> max{};
};

struct RoiConfig { Box3f box{{-10.F, -8.F, -3.F}, {50.F, 8.F, 3.F}}; };
struct EgoBoxConfig { Box3f box{{-1.5F, -1.7F, -1.F}, {2.6F, 1.7F, -0.4F}}; };
struct VoxelConfig { float leaf_size{0.2F}; std::uint32_t min_points_per_voxel{1}; };
struct GroundConfig {
  float max_angle_deg{15.F};
  float distance_threshold{0.2F};
  int max_iterations{100};
  double probability{0.99};
  bool optimize_coefficients{true};
  double min_inlier_ratio{0.15};
};
struct ClusterConfig { float tolerance{0.5F}; int min_points{10}; int max_points{1000}; };
struct ColorConfig {
  float min_distance{0.F};
  float max_distance{50.F};
  std::string palette{"red_green"};
  bool clamp{true};
  std::array<std::uint8_t, 3> ground{{120, 120, 120}};
  std::array<std::uint8_t, 3> unclustered{{80, 80, 80}};
  std::array<std::uint8_t, 3> ignored{{180, 180, 180}};
};
struct SaveConfig { bool binary_compressed{true}; };
struct PlaybackConfig { double fps{10.0}; bool loop{false}; };

struct PipelineConfig {
  RoiConfig roi;
  EgoBoxConfig ego;
  VoxelConfig voxel;
  GroundConfig ground;
  ClusterConfig cluster;
  ColorConfig color;
  SaveConfig save;
  PlaybackConfig playback;
};

struct Obstacle {
  std::uint32_t instance_id{0};
  std::size_t point_count{0};
  Box3f aabb;
  std::array<float, 3> centroid{};
  float sensor_distance{0.F};
  std::array<std::uint8_t, 3> color{};
  std::size_t min_original_index{0};
};

struct PointCounts {
  std::size_t input{0};
  std::size_t valid{0};
  std::size_t roi{0};
  std::size_t voxel{0};
  std::size_t ground_voxel{0};
  std::size_t non_ground_voxel{0};
  std::size_t obstacle_points{0};
};

struct StageTimings {
  double read_ms{0.0};
  double crop_ms{0.0};
  double voxel_ms{0.0};
  double ground_ms{0.0};
  double cluster_ms{0.0};
  double label_ms{0.0};
  double algorithm_ms{0.0};
  double save_ms{0.0};
};

struct DetectionResult {
  std::string frame_id;
  std::filesystem::path source_path;
  pcl::PointCloud<LabeledPoint>::Ptr labeled_cloud{new pcl::PointCloud<LabeledPoint>};
  std::array<float, 4> ground_coefficients{};
  std::vector<Obstacle> obstacles;
  PointCounts counts;
  StageTimings timings;
};

inline std::uint32_t packRgba(const std::array<std::uint8_t, 3>& rgb) {
  return (0xffU << 24U) | (static_cast<std::uint32_t>(rgb[0]) << 16U) |
         (static_cast<std::uint32_t>(rgb[1]) << 8U) | rgb[2];
}

}  // namespace mls::obstacle_detection

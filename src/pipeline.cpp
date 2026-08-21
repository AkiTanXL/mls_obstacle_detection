#include "mls/obstacle_detection/pipeline.hpp"
#include "mls/obstacle_detection/config.hpp"

#include <pcl/ModelCoefficients.h>
#include <pcl/common/point_tests.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mls::obstacle_detection {
namespace {

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

bool inside(const pcl::PointXYZI& point, const Box3f& box) {
  return point.x >= box.min[0] && point.x <= box.max[0] &&
         point.y >= box.min[1] && point.y <= box.max[1] &&
         point.z >= box.min[2] && point.z <= box.max[2];
}

std::vector<std::string> horizontalBoundaryFaces(const Box3f& aabb, const Box3f& roi,
                                                 float margin) {
  std::vector<std::string> faces;
  if (aabb.min[0] - roi.min[0] <= margin) faces.emplace_back("x_min");
  if (roi.max[0] - aabb.max[0] <= margin) faces.emplace_back("x_max");
  if (aabb.min[1] - roi.min[1] <= margin) faces.emplace_back("y_min");
  if (roi.max[1] - aabb.max[1] <= margin) faces.emplace_back("y_max");
  return faces;
}

struct ClusterCandidate {
  std::size_t raw_cluster_index{0};
  Obstacle obstacle;
  std::vector<std::string> boundary_faces;
};

std::array<std::uint8_t, 3> interpolate(const std::array<std::uint8_t, 3>& a,
                                        const std::array<std::uint8_t, 3>& b, float t) {
  std::array<std::uint8_t, 3> result{};
  for (std::size_t i = 0; i < result.size(); ++i) {
    result[i] = static_cast<std::uint8_t>(std::lround(a[i] + t * (static_cast<float>(b[i]) - a[i])));
  }
  return result;
}

}  // namespace

std::array<std::uint8_t, 3> distanceColor(float distance, const ColorConfig& config) {
  static constexpr std::array<std::array<std::uint8_t, 3>, 9> magma{{
      {{0, 0, 4}}, {{28, 16, 68}}, {{79, 18, 123}}, {{129, 37, 129}},
      {{181, 54, 122}}, {{229, 80, 100}}, {{251, 135, 97}}, {{254, 194, 135}},
      {{252, 253, 191}},
  }};
  float t = (distance - config.min_distance) / (config.max_distance - config.min_distance);
  t = std::clamp(t, 0.F, 1.F);
  if (config.palette == "red_green") {
    return interpolate({{255, 0, 0}}, {{0, 255, 0}}, t);
  }
  const float scaled = t * static_cast<float>(magma.size() - 1);
  const auto lower = static_cast<std::size_t>(std::floor(scaled));
  const auto upper = std::min(lower + 1, magma.size() - 1);
  return interpolate(magma[lower], magma[upper], scaled - static_cast<float>(lower));
}

ObstacleDetectionPipeline::ObstacleDetectionPipeline(PipelineConfig config) : config_(std::move(config)) {
  validateConfig(config_);
}

DetectionResult ObstacleDetectionPipeline::process(const Frame& frame) const {
  if (!frame.cloud) throw std::invalid_argument("frame cloud is null");
  const auto algorithm_begin = Clock::now();
  DetectionResult result;
  result.frame_id = frame.id;
  result.source_path = frame.source_path;
  result.counts.input = frame.cloud->size();
  result.labeled_cloud->header = frame.cloud->header;
  result.labeled_cloud->width = frame.cloud->width;
  result.labeled_cloud->height = frame.cloud->height;
  result.labeled_cloud->is_dense = frame.cloud->is_dense;
  result.labeled_cloud->sensor_origin_ = frame.sensor_origin;
  result.labeled_cloud->sensor_orientation_ = frame.sensor_orientation;
  result.labeled_cloud->points.resize(frame.cloud->size());
  for (std::size_t i = 0; i < frame.cloud->size(); ++i) {
    const auto& input = frame.cloud->points[i];
    auto& output = result.labeled_cloud->points[i];
    output.x = input.x;
    output.y = input.y;
    output.z = input.z;
    output.intensity = input.intensity;
    output.rgba = packRgba(config_.color.ignored);
    output.semantic_label = static_cast<std::uint32_t>(SemanticLabel::ignored);
    output.instance_id = 0;
  }

  const auto crop_begin = Clock::now();
  auto roi_cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  roi_cloud->reserve(frame.cloud->size());
  std::vector<std::size_t> roi_to_original;
  roi_to_original.reserve(frame.cloud->size());
  for (std::size_t i = 0; i < frame.cloud->size(); ++i) {
    const auto& point = frame.cloud->points[i];
    if (!pcl::isFinite(point)) continue;
    ++result.counts.valid;
    if (!inside(point, config_.roi.box) || inside(point, config_.ego.box)) continue;
    roi_cloud->push_back(point);
    roi_to_original.push_back(i);
    auto& labeled = result.labeled_cloud->points[i];
    labeled.semantic_label = static_cast<std::uint32_t>(SemanticLabel::non_ground_unclustered);
    labeled.rgba = packRgba(config_.color.unclustered);
  }
  result.counts.roi = roi_cloud->size();
  const auto crop_end = Clock::now();
  result.timings.crop_ms = elapsedMs(crop_begin, crop_end);
  if (roi_cloud->size() < 3) throw std::runtime_error("fewer than 3 valid points remain after ROI/ego cropping");

  const auto voxel_begin = Clock::now();
  pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
  voxel_filter.setInputCloud(roi_cloud);
  voxel_filter.setLeafSize(config_.voxel.leaf_size, config_.voxel.leaf_size, config_.voxel.leaf_size);
  voxel_filter.setMinimumPointsNumberPerVoxel(config_.voxel.min_points_per_voxel);
  voxel_filter.setSaveLeafLayout(true);
  auto voxel_cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  voxel_filter.filter(*voxel_cloud);
  result.counts.voxel = voxel_cloud->size();
  std::vector<int> roi_to_voxel(roi_cloud->size(), -1);
  for (std::size_t i = 0; i < roi_cloud->size(); ++i) {
    roi_to_voxel[i] = voxel_filter.getCentroidIndex(roi_cloud->points[i]);
  }
  const auto voxel_end = Clock::now();
  result.timings.voxel_ms = elapsedMs(voxel_begin, voxel_end);
  if (voxel_cloud->size() < 3) throw std::runtime_error("fewer than 3 voxels remain after downsampling");

  const auto ground_begin = Clock::now();
  pcl::SACSegmentation<pcl::PointXYZI> segmentation;
  segmentation.setOptimizeCoefficients(config_.ground.optimize_coefficients);
  segmentation.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
  segmentation.setMethodType(pcl::SAC_RANSAC);
  segmentation.setAxis(Eigen::Vector3f::UnitZ());
  segmentation.setEpsAngle(config_.ground.max_angle_deg * static_cast<float>(std::acos(-1.0) / 180.0));
  segmentation.setDistanceThreshold(config_.ground.distance_threshold);
  segmentation.setMaxIterations(config_.ground.max_iterations);
  segmentation.setProbability(config_.ground.probability);
  segmentation.setInputCloud(voxel_cloud);
  pcl::PointIndices ground_indices;
  pcl::ModelCoefficients coefficients;
  segmentation.segment(ground_indices, coefficients);
  if (ground_indices.indices.empty() || coefficients.values.size() < 4) {
    throw std::runtime_error("no valid constrained ground plane found");
  }
  const double ground_ratio = static_cast<double>(ground_indices.indices.size()) /
                              static_cast<double>(voxel_cloud->size());
  if (ground_ratio < config_.ground.min_inlier_ratio) {
    throw std::runtime_error("ground plane inlier ratio is below configured minimum");
  }
  std::copy_n(coefficients.values.begin(), 4, result.ground_coefficients.begin());
  std::vector<bool> is_ground(voxel_cloud->size(), false);
  for (int index : ground_indices.indices) {
    if (index >= 0 && static_cast<std::size_t>(index) < is_ground.size()) is_ground[static_cast<std::size_t>(index)] = true;
  }
  result.counts.ground_voxel = ground_indices.indices.size();
  result.counts.non_ground_voxel = voxel_cloud->size() - result.counts.ground_voxel;
  const auto ground_end = Clock::now();
  result.timings.ground_ms = elapsedMs(ground_begin, ground_end);

  const auto cluster_begin = Clock::now();
  auto non_ground_cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  std::vector<std::size_t> cluster_to_voxel;
  non_ground_cloud->reserve(result.counts.non_ground_voxel);
  cluster_to_voxel.reserve(result.counts.non_ground_voxel);
  for (std::size_t i = 0; i < voxel_cloud->size(); ++i) {
    if (!is_ground[i]) {
      non_ground_cloud->push_back(voxel_cloud->points[i]);
      cluster_to_voxel.push_back(i);
    }
  }
  std::vector<pcl::PointIndices> clusters;
  if (!non_ground_cloud->empty()) {
    auto tree = pcl::search::KdTree<pcl::PointXYZI>::Ptr(new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(non_ground_cloud);
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> extraction;
    extraction.setClusterTolerance(config_.cluster.tolerance);
    extraction.setMinClusterSize(config_.cluster.min_points);
    extraction.setMaxClusterSize(config_.cluster.max_points);
    extraction.setSearchMethod(tree);
    extraction.setInputCloud(non_ground_cloud);
    extraction.extract(clusters);
  }
  std::vector<int> voxel_to_cluster(voxel_cloud->size(), -1);
  for (std::size_t cluster = 0; cluster < clusters.size(); ++cluster) {
    for (int index : clusters[cluster].indices) {
      voxel_to_cluster[cluster_to_voxel.at(static_cast<std::size_t>(index))] = static_cast<int>(cluster);
    }
  }
  const auto cluster_end = Clock::now();
  result.timings.cluster_ms = elapsedMs(cluster_begin, cluster_end);

  const auto label_begin = Clock::now();
  std::vector<std::vector<std::size_t>> raw_cluster_indices(clusters.size());
  for (std::size_t roi_index = 0; roi_index < roi_to_original.size(); ++roi_index) {
    const std::size_t original_index = roi_to_original[roi_index];
    const int voxel_index = roi_to_voxel[roi_index];
    auto& output = result.labeled_cloud->points[original_index];
    if (voxel_index < 0) continue;
    if (is_ground.at(static_cast<std::size_t>(voxel_index))) {
      output.semantic_label = static_cast<std::uint32_t>(SemanticLabel::ground);
      output.rgba = packRgba(config_.color.ground);
      continue;
    }
    const int cluster = voxel_to_cluster.at(static_cast<std::size_t>(voxel_index));
    if (cluster >= 0) raw_cluster_indices.at(static_cast<std::size_t>(cluster)).push_back(original_index);
  }

  std::vector<ClusterCandidate> candidates;
  candidates.reserve(raw_cluster_indices.size());
  for (std::size_t cluster = 0; cluster < raw_cluster_indices.size(); ++cluster) {
    const auto& indices = raw_cluster_indices[cluster];
    if (indices.empty()) continue;
    ClusterCandidate candidate;
    candidate.raw_cluster_index = cluster;
    auto& obstacle = candidate.obstacle;
    obstacle.point_count = indices.size();
    obstacle.min_original_index = *std::min_element(indices.begin(), indices.end());
    obstacle.aabb.min.fill(std::numeric_limits<float>::infinity());
    obstacle.aabb.max.fill(-std::numeric_limits<float>::infinity());
    std::array<double, 3> sum{};
    for (const std::size_t index : indices) {
      const auto& point = frame.cloud->points[index];
      const std::array<float, 3> xyz{{point.x, point.y, point.z}};
      for (std::size_t axis = 0; axis < 3; ++axis) {
        obstacle.aabb.min[axis] = std::min(obstacle.aabb.min[axis], xyz[axis]);
        obstacle.aabb.max[axis] = std::max(obstacle.aabb.max[axis], xyz[axis]);
        sum[axis] += xyz[axis];
      }
    }
    for (std::size_t axis = 0; axis < 3; ++axis) {
      obstacle.centroid[axis] = static_cast<float>(sum[axis] / static_cast<double>(indices.size()));
    }
    const float dx = 0.5F * (obstacle.aabb.min[0] + obstacle.aabb.max[0]) - frame.sensor_origin[0];
    const float dy = 0.5F * (obstacle.aabb.min[1] + obstacle.aabb.max[1]) - frame.sensor_origin[1];
    const float dz = 0.5F * (obstacle.aabb.min[2] + obstacle.aabb.max[2]) - frame.sensor_origin[2];
    obstacle.sensor_distance = std::sqrt(dx * dx + dy * dy + dz * dz);
    obstacle.color = distanceColor(obstacle.sensor_distance, config_.color);
    if (config_.cluster_filter.enabled) {
      candidate.boundary_faces = horizontalBoundaryFaces(
          obstacle.aabb, config_.roi.box, config_.cluster_filter.roi_boundary_margin);
    }
    candidates.push_back(std::move(candidate));
  }
  std::stable_sort(candidates.begin(), candidates.end(), [](const ClusterCandidate& a,
                                                             const ClusterCandidate& b) {
    return std::tie(a.obstacle.sensor_distance, a.obstacle.centroid[0], a.obstacle.centroid[1],
                    a.obstacle.centroid[2], a.obstacle.min_original_index) <
           std::tie(b.obstacle.sensor_distance, b.obstacle.centroid[0], b.obstacle.centroid[1],
                    b.obstacle.centroid[2], b.obstacle.min_original_index);
  });
  for (auto& candidate : candidates) {
    const auto& indices = raw_cluster_indices[candidate.raw_cluster_index];
    const bool filtered = config_.cluster_filter.enabled && !candidate.boundary_faces.empty();
    if (filtered) {
      FilteredCluster diagnostic;
      diagnostic.point_count = candidate.obstacle.point_count;
      diagnostic.aabb = candidate.obstacle.aabb;
      diagnostic.boundary_faces = std::move(candidate.boundary_faces);
      result.filtered_clusters.push_back(std::move(diagnostic));
      result.counts.filtered_cluster_points += indices.size();
      continue;
    }

    auto& obstacle = candidate.obstacle;
    obstacle.instance_id = static_cast<std::uint32_t>(result.obstacles.size() + 1);
    for (const std::size_t index : indices) {
      auto& output = result.labeled_cloud->points[index];
      output.semantic_label = static_cast<std::uint32_t>(SemanticLabel::obstacle);
      output.instance_id = obstacle.instance_id;
      output.rgba = packRgba(obstacle.color);
    }
    result.counts.obstacle_points += indices.size();
    result.obstacles.push_back(std::move(obstacle));
  }
  const auto label_end = Clock::now();
  result.timings.label_ms = elapsedMs(label_begin, label_end);
  result.timings.algorithm_ms = elapsedMs(algorithm_begin, label_end);
  return result;
}

}  // namespace mls::obstacle_detection

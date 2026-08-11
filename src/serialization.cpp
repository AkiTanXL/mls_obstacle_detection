#include "mls/obstacle_detection/serialization.hpp"

#include <nlohmann/json.hpp>
#include <pcl/io/pcd_io.h>

#include <chrono>
#include <fstream>
#include <stdexcept>
#include <system_error>

namespace mls::obstacle_detection {
namespace {

std::filesystem::path temporaryPath(const std::filesystem::path& target) {
  const auto token = std::chrono::steady_clock::now().time_since_epoch().count();
  return target.parent_path() / (target.filename().string() + ".tmp." + std::to_string(token));
}

void replaceWith(const std::filesystem::path& temporary, const std::filesystem::path& target) {
  std::error_code error;
  std::filesystem::rename(temporary, target, error);
  if (error) throw std::runtime_error("cannot atomically replace '" + target.string() + "': " + error.message());
}

nlohmann::json boxJson(const Box3f& box) {
  return {{"min", box.min}, {"max", box.max}};
}

}  // namespace

void prepareOutputDirectory(const std::filesystem::path& output, bool overwrite) {
  std::error_code error;
  if (std::filesystem::exists(output, error)) {
    if (!std::filesystem::is_directory(output, error)) {
      throw std::runtime_error("output path exists and is not a directory: " + output.string());
    }
    if (!overwrite && std::filesystem::directory_iterator(output) != std::filesystem::directory_iterator()) {
      throw std::runtime_error("output directory is not empty (use --overwrite): " + output.string());
    }
  } else if (!std::filesystem::create_directories(output, error) && error) {
    throw std::runtime_error("cannot create output directory '" + output.string() + "': " + error.message());
  }
}

OutputPaths outputPathsFor(const std::filesystem::path& output, const Frame& frame) {
  return {output / (frame.id + ".pcd"), output / (frame.id + ".json")};
}

std::string resultToJson(const Frame& frame, const DetectionResult& result) {
  nlohmann::json obstacles = nlohmann::json::array();
  for (const auto& obstacle : result.obstacles) {
    obstacles.push_back({
        {"instance_id", obstacle.instance_id},
        {"point_count", obstacle.point_count},
        {"aabb", boxJson(obstacle.aabb)},
        {"centroid", obstacle.centroid},
        {"sensor_distance", obstacle.sensor_distance},
        {"color_rgb", obstacle.color},
    });
  }
  const double crop_ratio = result.counts.valid == 0 ? 0.0 :
      static_cast<double>(result.counts.roi) / static_cast<double>(result.counts.valid);
  const double ground_ratio = result.counts.voxel == 0 ? 0.0 :
      static_cast<double>(result.counts.ground_voxel) / static_cast<double>(result.counts.voxel);
  std::array<std::size_t, 4> raw_labels{};
  for (const auto& point : result.labeled_cloud->points) {
    if (point.semantic_label < raw_labels.size()) ++raw_labels[point.semantic_label];
  }
  nlohmann::json json{
      {"schema_version", 1},
      {"frame_id", result.frame_id},
      {"source", frame.source_path.string()},
      {"point_counts", {
          {"input", result.counts.input}, {"valid", result.counts.valid},
          {"roi", result.counts.roi}, {"voxel", result.counts.voxel},
          {"ground_voxel", result.counts.ground_voxel},
          {"non_ground_voxel", result.counts.non_ground_voxel},
          {"ignored_raw", raw_labels[0]}, {"ground_raw", raw_labels[1]},
          {"unclustered_raw", raw_labels[2]}, {"obstacle_raw", raw_labels[3]}}},
      {"ratios", {{"crop_retained", crop_ratio}, {"ground_voxel", ground_ratio}}},
      {"ground_coefficients", result.ground_coefficients},
      {"timings_ms", {
          {"read", result.timings.read_ms}, {"crop", result.timings.crop_ms},
          {"voxel", result.timings.voxel_ms}, {"ground", result.timings.ground_ms},
          {"cluster", result.timings.cluster_ms}, {"label", result.timings.label_ms},
          {"algorithm", result.timings.algorithm_ms}, {"save", result.timings.save_ms}}},
      {"obstacle_count", result.obstacles.size()},
      {"obstacles", std::move(obstacles)},
  };
  return json.dump(2) + "\n";
}

void writeTextAtomic(const std::filesystem::path& path, const std::string& text) {
  const auto temporary = temporaryPath(path);
  try {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open temporary output: " + temporary.string());
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    output.flush();
    if (!output) throw std::runtime_error("failed writing temporary output: " + temporary.string());
    output.close();
    replaceWith(temporary, path);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

void writeFrameResultAtomic(const OutputPaths& paths, const Frame& frame,
                            DetectionResult& result, bool binary_compressed) {
  const auto save_begin = std::chrono::steady_clock::now();
  const auto pcd_temporary = temporaryPath(paths.pcd);
  const auto json_temporary = temporaryPath(paths.json);
  try {
    const int status = binary_compressed
        ? pcl::io::savePCDFileBinaryCompressed(pcd_temporary.string(), *result.labeled_cloud)
        : pcl::io::savePCDFileBinary(pcd_temporary.string(), *result.labeled_cloud);
    if (status < 0) throw std::runtime_error("failed to write PCD temporary file: " + pcd_temporary.string());
    result.timings.save_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - save_begin).count();
    {
      std::ofstream output(json_temporary, std::ios::binary | std::ios::trunc);
      if (!output) throw std::runtime_error("cannot open JSON temporary file: " + json_temporary.string());
      output << resultToJson(frame, result);
      output.flush();
      if (!output) throw std::runtime_error("failed writing JSON temporary file: " + json_temporary.string());
    }
    replaceWith(pcd_temporary, paths.pcd);
    replaceWith(json_temporary, paths.json);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(pcd_temporary, ignored);
    std::filesystem::remove(json_temporary, ignored);
    throw;
  }
}

}  // namespace mls::obstacle_detection

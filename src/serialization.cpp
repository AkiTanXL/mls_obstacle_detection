#include "mls/obstacle_detection/serialization.hpp"

#include <nlohmann/json.hpp>
#include <pcl/io/pcd_io.h>

#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

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

struct AsyncFrameWriter::Impl {
  struct SaveJob {
    OutputPaths paths;
    DetectionResult result;
  };

  explicit Impl(bool binary_compressed_value, std::size_t capacity_value)
      : binary_compressed(binary_compressed_value), capacity(capacity_value) {
    if (capacity == 0) throw std::invalid_argument("async writer capacity must be positive");
    worker = std::thread([this] { run(); });
  }

  ~Impl() { closeAndJoin(); }

  void enqueue(OutputPaths paths, DetectionResult result) {
    std::unique_lock<std::mutex> lock(mutex);
    space_available.wait(lock, [this] {
      return pending < capacity || error != nullptr || closing;
    });
    if (error) {
      const auto saved_error = error;
      lock.unlock();
      std::rethrow_exception(saved_error);
    }
    if (closing) throw std::logic_error("cannot enqueue after async writer is closed");
    jobs.push_back({std::move(paths), std::move(result)});
    ++pending;
    lock.unlock();
    work_ready.notify_one();
  }

  std::vector<FrameSaveCompletion> takeCompletions() {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<FrameSaveCompletion> completed;
    completed.swap(completions);
    return completed;
  }

  void rethrowIfFailed() const {
    std::exception_ptr saved_error;
    {
      std::lock_guard<std::mutex> lock(mutex);
      saved_error = error;
    }
    if (saved_error) std::rethrow_exception(saved_error);
  }

  void finish() {
    closeAndJoin();
    rethrowIfFailed();
  }

  void closeAndJoin() noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex);
      closing = true;
    }
    work_ready.notify_all();
    space_available.notify_all();
    if (worker.joinable()) worker.join();
  }

  void run() noexcept {
    while (true) {
      SaveJob job;
      {
        std::unique_lock<std::mutex> lock(mutex);
        work_ready.wait(lock, [this] { return closing || !jobs.empty(); });
        if (jobs.empty()) return;
        job = std::move(jobs.front());
        jobs.pop_front();
      }
      try {
        writeFrameResultAtomic(job.paths, job.result, binary_compressed);
        FrameSaveCompletion completion{
            job.result.source_path, job.result.counts, job.result.obstacles.size(),
            job.result.filtered_clusters.size(), job.result.timings.algorithm_ms,
            job.result.timings.save_ms};
        {
          std::lock_guard<std::mutex> lock(mutex);
          completions.push_back(std::move(completion));
          --pending;
        }
        space_available.notify_all();
      } catch (...) {
        {
          std::lock_guard<std::mutex> lock(mutex);
          error = std::current_exception();
          jobs.clear();
          pending = 0;
          closing = true;
        }
        space_available.notify_all();
        return;
      }
    }
  }

  bool binary_compressed{true};
  std::size_t capacity{2};
  mutable std::mutex mutex;
  std::condition_variable work_ready;
  std::condition_variable space_available;
  std::deque<SaveJob> jobs;
  std::vector<FrameSaveCompletion> completions;
  std::size_t pending{0};
  bool closing{false};
  std::exception_ptr error;
  std::thread worker;
};

AsyncFrameWriter::AsyncFrameWriter(bool binary_compressed, std::size_t capacity)
    : impl_(std::make_unique<Impl>(binary_compressed, capacity)) {}

AsyncFrameWriter::~AsyncFrameWriter() = default;

void AsyncFrameWriter::enqueue(OutputPaths paths, DetectionResult result) {
  impl_->enqueue(std::move(paths), std::move(result));
}

std::vector<FrameSaveCompletion> AsyncFrameWriter::takeCompletions() {
  return impl_->takeCompletions();
}

void AsyncFrameWriter::rethrowIfFailed() const { impl_->rethrowIfFailed(); }

void AsyncFrameWriter::finish() { impl_->finish(); }

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

std::string resultToJson(const DetectionResult& result) {
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
  nlohmann::json filtered_clusters = nlohmann::json::array();
  for (const auto& cluster : result.filtered_clusters) {
    filtered_clusters.push_back({
        {"point_count", cluster.point_count},
        {"aabb", boxJson(cluster.aabb)},
        {"boundary_faces", cluster.boundary_faces},
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
      {"schema_version", 3},
      {"frame_id", result.frame_id},
      {"source", result.source_path.string()},
      {"point_counts", {
          {"input", result.counts.input}, {"valid", result.counts.valid},
          {"roi", result.counts.roi}, {"voxel", result.counts.voxel},
          {"ground_voxel", result.counts.ground_voxel},
          {"non_ground_voxel", result.counts.non_ground_voxel},
          {"filtered_cluster_raw", result.counts.filtered_cluster_points},
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
      {"filtered_cluster_count", result.filtered_clusters.size()},
      {"filtered_clusters", std::move(filtered_clusters)},
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

void writeFrameResultAtomic(const OutputPaths& paths, DetectionResult& result,
                            bool binary_compressed) {
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
      output << resultToJson(result);
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

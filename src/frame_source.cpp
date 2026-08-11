#include "mls/obstacle_detection/frame_source.hpp"

#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mls::obstacle_detection {
namespace {

bool allDigits(std::string_view value) {
  return !value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char c) { return std::isdigit(c); });
}

std::string_view normalizedNumber(std::string_view value) {
  const auto first = value.find_first_not_of('0');
  return first == std::string_view::npos ? value.substr(value.size() - 1) : value.substr(first);
}

bool pathLess(const std::filesystem::path& lhs, const std::filesystem::path& rhs) {
  const std::string a = lhs.stem().string();
  const std::string b = rhs.stem().string();
  const bool an = allDigits(a);
  const bool bn = allDigits(b);
  if (an && bn) {
    const auto av = normalizedNumber(a);
    const auto bv = normalizedNumber(b);
    if (av.size() != bv.size()) return av.size() < bv.size();
    if (av != bv) return av < bv;
    if (a.size() != b.size()) return a.size() < b.size();
  } else if (an != bn) {
    return an;
  }
  return lhs.filename().string() < rhs.filename().string();
}

bool hasField(const pcl::PCLPointCloud2& cloud, const std::string& name) {
  return std::any_of(cloud.fields.begin(), cloud.fields.end(), [&](const pcl::PCLPointField& field) {
    return field.name == name;
  });
}

std::vector<std::filesystem::path> collectPaths(
    const std::filesystem::path& input, std::string_view extension) {
  std::vector<std::filesystem::path> paths;
  std::error_code error;
  if (std::filesystem::is_regular_file(input, error)) {
    if (input.extension() != extension) {
      throw std::invalid_argument("input file must have " + std::string(extension) +
                                  " extension");
    }
    paths.push_back(std::filesystem::absolute(input));
  } else if (std::filesystem::is_directory(input, error)) {
    for (const auto& entry : std::filesystem::directory_iterator(input)) {
      if (entry.is_regular_file() && entry.path().extension() == extension) {
        paths.push_back(std::filesystem::absolute(entry.path()));
      }
    }
    std::sort(paths.begin(), paths.end(), pathLess);
    if (paths.empty()) {
      throw std::invalid_argument("input directory contains no direct-child " +
                                  std::string(extension) + " files");
    }
  } else {
    throw std::invalid_argument("input does not exist or is not a file/directory: " +
                                input.string());
  }
  return paths;
}

float littleEndianFloat(const char* bytes) {
  static_assert(sizeof(float) == sizeof(std::uint32_t),
                "KITTI BIN loading requires 32-bit float");
  static_assert(std::numeric_limits<float>::is_iec559,
                "KITTI BIN loading requires IEEE-754 float");
  const auto* data = reinterpret_cast<const unsigned char*>(bytes);
  const std::uint32_t bits = static_cast<std::uint32_t>(data[0]) |
                             (static_cast<std::uint32_t>(data[1]) << 8U) |
                             (static_cast<std::uint32_t>(data[2]) << 16U) |
                             (static_cast<std::uint32_t>(data[3]) << 24U);
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

}  // namespace

PcdFrameSource::PcdFrameSource(const std::filesystem::path& input)
    : paths_(collectPaths(input, ".pcd")) {}

bool PcdFrameSource::next(Frame& frame) {
  if (cursor_ >= paths_.size()) return false;
  const auto path = paths_[cursor_++];
  pcl::PCLPointCloud2 blob;
  Eigen::Vector4f origin = Eigen::Vector4f::Zero();
  Eigen::Quaternionf orientation = Eigen::Quaternionf::Identity();
  if (pcl::io::loadPCDFile(path.string(), blob, origin, orientation) < 0) {
    throw std::runtime_error("failed to read PCD: " + path.string());
  }
  for (const char* field : {"x", "y", "z"}) {
    if (!hasField(blob, field)) throw std::runtime_error("PCD is missing required field '" + std::string(field) + "': " + path.string());
  }
  const bool has_intensity = hasField(blob, "intensity");
  auto cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  if (has_intensity) {
    pcl::fromPCLPointCloud2(blob, *cloud);
  } else {
    pcl::PointCloud<pcl::PointXYZ> xyz;
    pcl::fromPCLPointCloud2(blob, xyz);
    cloud->header = xyz.header;
    cloud->width = xyz.width;
    cloud->height = xyz.height;
    cloud->is_dense = xyz.is_dense;
    cloud->points.resize(xyz.size());
    for (std::size_t i = 0; i < xyz.size(); ++i) {
      cloud->points[i].x = xyz.points[i].x;
      cloud->points[i].y = xyz.points[i].y;
      cloud->points[i].z = xyz.points[i].z;
      cloud->points[i].intensity = 0.F;
    }
  }
  cloud->sensor_origin_ = origin;
  cloud->sensor_orientation_ = orientation;

  frame.id = path.stem().string();
  frame.source_path = path;
  frame.cloud = std::move(cloud);
  frame.sensor_origin = origin;
  frame.sensor_orientation = orientation;
  return true;
}

void PcdFrameSource::reset() { cursor_ = 0; }
std::size_t PcdFrameSource::size() const { return paths_.size(); }
const std::vector<std::filesystem::path>& PcdFrameSource::paths() const { return paths_; }

BinFrameSource::BinFrameSource(const std::filesystem::path& input)
    : paths_(collectPaths(input, ".bin")) {}

bool BinFrameSource::next(Frame& frame) {
  if (cursor_ >= paths_.size()) return false;
  const auto path = paths_[cursor_++];
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("failed to open KITTI BIN: " + path.string());
  const auto end = input.tellg();
  if (end < 0) throw std::runtime_error("failed to determine KITTI BIN size: " + path.string());
  const auto byte_count = static_cast<std::uintmax_t>(end);
  constexpr std::uintmax_t record_size = 4U * sizeof(float);
  if (byte_count == 0U) throw std::runtime_error("KITTI BIN contains no points: " + path.string());
  if (byte_count % record_size != 0U) {
    throw std::runtime_error("KITTI BIN size is not a multiple of 16 bytes: " + path.string());
  }
  if (byte_count > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max()) ||
      byte_count > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("KITTI BIN is too large to read: " + path.string());
  }
  const std::size_t point_count = static_cast<std::size_t>(byte_count / record_size);
  if (point_count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("KITTI BIN has too many points for PCL: " + path.string());
  }

  std::vector<char> bytes(static_cast<std::size_t>(byte_count));
  input.seekg(0);
  input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!input) throw std::runtime_error("failed to read complete KITTI BIN: " + path.string());

  auto cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  cloud->points.resize(point_count);
  cloud->width = static_cast<std::uint32_t>(point_count);
  cloud->height = 1U;
  cloud->is_dense = true;
  for (std::size_t i = 0; i < point_count; ++i) {
    const char* record = bytes.data() + i * static_cast<std::size_t>(record_size);
    auto& point = cloud->points[i];
    point.x = littleEndianFloat(record);
    point.y = littleEndianFloat(record + sizeof(float));
    point.z = littleEndianFloat(record + 2U * sizeof(float));
    point.intensity = littleEndianFloat(record + 3U * sizeof(float));
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      cloud->is_dense = false;
    }
  }
  cloud->sensor_origin_ = Eigen::Vector4f::Zero();
  cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();

  frame.id = path.stem().string();
  frame.source_path = path;
  frame.cloud = std::move(cloud);
  frame.sensor_origin = Eigen::Vector4f::Zero();
  frame.sensor_orientation = Eigen::Quaternionf::Identity();
  return true;
}

void BinFrameSource::reset() { cursor_ = 0; }
std::size_t BinFrameSource::size() const { return paths_.size(); }
const std::vector<std::filesystem::path>& BinFrameSource::paths() const { return paths_; }

std::unique_ptr<FrameSource> makeFrameSource(const std::filesystem::path& input) {
  std::error_code error;
  if (std::filesystem::is_regular_file(input, error)) {
    const auto extension = input.extension();
    if (extension == ".pcd") return std::make_unique<PcdFrameSource>(input);
    if (extension == ".bin") return std::make_unique<BinFrameSource>(input);
    throw std::invalid_argument("unsupported input file extension '" + extension.string() +
                                "'; expected .pcd or .bin");
  }
  if (std::filesystem::is_directory(input, error)) {
    bool has_pcd = false;
    bool has_bin = false;
    for (const auto& entry : std::filesystem::directory_iterator(input)) {
      if (!entry.is_regular_file()) continue;
      has_pcd = has_pcd || entry.path().extension() == ".pcd";
      has_bin = has_bin || entry.path().extension() == ".bin";
    }
    if (has_pcd && has_bin) {
      throw std::invalid_argument(
          "input directory mixes direct-child .pcd and .bin files; split formats into separate directories");
    }
    if (has_pcd) return std::make_unique<PcdFrameSource>(input);
    if (has_bin) return std::make_unique<BinFrameSource>(input);
    throw std::invalid_argument(
        "input directory contains no direct-child .pcd or .bin files");
  }
  throw std::invalid_argument("input does not exist or is not a file/directory: " +
                              input.string());
}

}  // namespace mls::obstacle_detection

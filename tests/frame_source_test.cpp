#include "mls/obstacle_detection/frame_source.hpp"
#include "test_support.hpp"

#include <gtest/gtest.h>
#include <pcl/io/pcd_io.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <limits>

namespace od = mls::obstacle_detection;

namespace {

void writeLittleEndianFloat(std::ofstream& output, float value) {
  static_assert(sizeof(float) == sizeof(std::uint32_t));
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::array<char, 4> bytes{
      static_cast<char>(bits & 0xffU),
      static_cast<char>((bits >> 8U) & 0xffU),
      static_cast<char>((bits >> 16U) & 0xffU),
      static_cast<char>((bits >> 24U) & 0xffU)};
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void writeKittiBin(
    const std::filesystem::path& path,
    std::initializer_list<std::array<float, 4>> points) {
  std::ofstream output(path, std::ios::binary);
  for (const auto& point : points) {
    for (float value : point) writeLittleEndianFloat(output, value);
  }
}

}  // namespace

TEST(PcdFrameSource, SortsNumericStemsNaturally) {
  TemporaryDirectory temporary;
  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.push_back({1.F, 2.F, 3.F});
  for (const char* name : {"10.pcd", "2.pcd", "001.pcd", "alpha.pcd"}) {
    ASSERT_EQ(pcl::io::savePCDFileBinary((temporary.path() / name).string(), cloud), 0);
  }
  std::filesystem::create_directories(temporary.path() / "nested");
  ASSERT_EQ(pcl::io::savePCDFileBinary((temporary.path() / "nested/0.pcd").string(), cloud), 0);
  od::PcdFrameSource source(temporary.path());
  ASSERT_EQ(source.size(), 4U);
  EXPECT_EQ(source.paths()[0].filename(), "001.pcd");
  EXPECT_EQ(source.paths()[1].filename(), "2.pcd");
  EXPECT_EQ(source.paths()[2].filename(), "10.pcd");
  EXPECT_EQ(source.paths()[3].filename(), "alpha.pcd");
}

TEST(PcdFrameSource, LoadsXyzAndPreservesOrganizationViewpointAndInvalidPoints) {
  TemporaryDirectory temporary;
  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.width = 2;
  cloud.height = 2;
  cloud.is_dense = false;
  cloud.points.resize(4);
  cloud.points[0] = {1.F, 2.F, 3.F};
  cloud.points[1] = {4.F, 5.F, 6.F};
  cloud.points[2] = {7.F, 8.F, 9.F};
  cloud.points[3] = {std::numeric_limits<float>::quiet_NaN(), 0.F, 0.F};
  cloud.sensor_origin_ = Eigen::Vector4f(1.F, 2.F, 3.F, 0.F);
  cloud.sensor_orientation_ = Eigen::Quaternionf(Eigen::AngleAxisf(0.2F, Eigen::Vector3f::UnitZ()));
  const auto path = temporary.path() / "xyz.pcd";
  ASSERT_EQ(pcl::io::savePCDFileBinary(path.string(), cloud), 0);
  od::PcdFrameSource source(path);
  od::Frame frame;
  ASSERT_TRUE(source.next(frame));
  EXPECT_EQ(frame.cloud->width, 2U);
  EXPECT_EQ(frame.cloud->height, 2U);
  ASSERT_EQ(frame.cloud->size(), 4U);
  for (const auto& point : frame.cloud->points) EXPECT_FLOAT_EQ(point.intensity, 0.F);
  EXPECT_TRUE(std::isnan(frame.cloud->points[3].x));
  EXPECT_TRUE(frame.sensor_origin.isApprox(cloud.sensor_origin_));
  EXPECT_TRUE(frame.sensor_orientation.isApprox(cloud.sensor_orientation_));
}

TEST(PcdFrameSource, KeepsIntensity) {
  TemporaryDirectory temporary;
  pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl::PointXYZI point;
  point.x = 1.F; point.y = 2.F; point.z = 3.F; point.intensity = 42.F;
  cloud.push_back(point);
  const auto path = temporary.path() / "xyzi.pcd";
  ASSERT_EQ(pcl::io::savePCDFileBinary(path.string(), cloud), 0);
  od::PcdFrameSource source(path);
  od::Frame frame;
  ASSERT_TRUE(source.next(frame));
  EXPECT_FLOAT_EQ(frame.cloud->front().intensity, 42.F);
}

TEST(BinFrameSource, SortsNumericStemsNaturally) {
  TemporaryDirectory temporary;
  for (const char* name : {"10.bin", "2.bin", "001.bin", "alpha.bin"}) {
    writeKittiBin(temporary.path() / name, {{1.F, 2.F, 3.F, 4.F}});
  }
  std::filesystem::create_directories(temporary.path() / "nested");
  writeKittiBin(temporary.path() / "nested/0.bin", {{1.F, 2.F, 3.F, 4.F}});
  { std::ofstream ignored(temporary.path() / "notes.txt"); ignored << "metadata"; }

  od::BinFrameSource source(temporary.path());
  ASSERT_EQ(source.size(), 4U);
  EXPECT_EQ(source.paths()[0].filename(), "001.bin");
  EXPECT_EQ(source.paths()[1].filename(), "2.bin");
  EXPECT_EQ(source.paths()[2].filename(), "10.bin");
  EXPECT_EQ(source.paths()[3].filename(), "alpha.bin");
}

TEST(BinFrameSource, LoadsKittiXyziAndSupportsReset) {
  TemporaryDirectory temporary;
  const auto path = temporary.path() / "000123.bin";
  writeKittiBin(path, {{1.25F, -2.5F, 3.75F, 0.5F},
                       {std::numeric_limits<float>::quiet_NaN(), 5.F, 6.F, 0.75F}});

  od::BinFrameSource source(path);
  od::Frame frame;
  ASSERT_TRUE(source.next(frame));
  EXPECT_EQ(frame.id, "000123");
  EXPECT_EQ(frame.source_path, std::filesystem::absolute(path));
  EXPECT_EQ(frame.cloud->width, 2U);
  EXPECT_EQ(frame.cloud->height, 1U);
  EXPECT_FALSE(frame.cloud->is_dense);
  ASSERT_EQ(frame.cloud->size(), 2U);
  EXPECT_FLOAT_EQ(frame.cloud->points[0].x, 1.25F);
  EXPECT_FLOAT_EQ(frame.cloud->points[0].y, -2.5F);
  EXPECT_FLOAT_EQ(frame.cloud->points[0].z, 3.75F);
  EXPECT_FLOAT_EQ(frame.cloud->points[0].intensity, 0.5F);
  EXPECT_TRUE(std::isnan(frame.cloud->points[1].x));
  EXPECT_TRUE(frame.sensor_origin.isApprox(Eigen::Vector4f::Zero()));
  EXPECT_TRUE(frame.sensor_orientation.isApprox(Eigen::Quaternionf::Identity()));
  EXPECT_FALSE(source.next(frame));
  source.reset();
  EXPECT_TRUE(source.next(frame));
}

TEST(BinFrameSource, RejectsEmptyAndMisalignedFiles) {
  TemporaryDirectory temporary;
  const auto empty_path = temporary.path() / "empty.bin";
  { std::ofstream empty(empty_path, std::ios::binary); }
  od::BinFrameSource empty_source(empty_path);
  od::Frame frame;
  EXPECT_THROW(empty_source.next(frame), std::runtime_error);

  const auto malformed_path = temporary.path() / "malformed.bin";
  {
    std::ofstream malformed(malformed_path, std::ios::binary);
    const std::array<char, 15> bytes{};
    malformed.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  od::BinFrameSource malformed_source(malformed_path);
  EXPECT_THROW(malformed_source.next(frame), std::runtime_error);
}

TEST(FrameSourceFactory, SelectsByFileOrDirectoryAndRejectsAmbiguity) {
  TemporaryDirectory temporary;
  const auto pcd_directory = temporary.path() / "pcd";
  const auto bin_directory = temporary.path() / "bin";
  const auto mixed_directory = temporary.path() / "mixed";
  const auto empty_directory = temporary.path() / "empty";
  std::filesystem::create_directories(pcd_directory);
  std::filesystem::create_directories(bin_directory);
  std::filesystem::create_directories(mixed_directory);
  std::filesystem::create_directories(empty_directory);

  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.push_back({1.F, 2.F, 3.F});
  const auto pcd_path = pcd_directory / "0.pcd";
  ASSERT_EQ(pcl::io::savePCDFileBinary(pcd_path.string(), cloud), 0);
  const auto bin_path = bin_directory / "0.bin";
  writeKittiBin(bin_path, {{1.F, 2.F, 3.F, 4.F}});
  ASSERT_EQ(pcl::io::savePCDFileBinary((mixed_directory / "0.pcd").string(), cloud), 0);
  writeKittiBin(mixed_directory / "1.bin", {{1.F, 2.F, 3.F, 4.F}});
  { std::ofstream ignored(empty_directory / "notes.txt"); ignored << "metadata"; }

  auto pcd_file_source = od::makeFrameSource(pcd_path);
  auto bin_file_source = od::makeFrameSource(bin_path);
  auto pcd_directory_source = od::makeFrameSource(pcd_directory);
  auto bin_directory_source = od::makeFrameSource(bin_directory);
  EXPECT_NE(dynamic_cast<od::PcdFrameSource*>(pcd_file_source.get()), nullptr);
  EXPECT_NE(dynamic_cast<od::BinFrameSource*>(bin_file_source.get()), nullptr);
  EXPECT_NE(dynamic_cast<od::PcdFrameSource*>(pcd_directory_source.get()), nullptr);
  EXPECT_NE(dynamic_cast<od::BinFrameSource*>(bin_directory_source.get()), nullptr);
  EXPECT_THROW({ auto source = od::makeFrameSource(mixed_directory); (void)source; },
               std::invalid_argument);
  EXPECT_THROW({ auto source = od::makeFrameSource(empty_directory); (void)source; },
               std::invalid_argument);
  EXPECT_THROW({
    auto source = od::makeFrameSource(empty_directory / "notes.txt");
    (void)source;
  }, std::invalid_argument);
}

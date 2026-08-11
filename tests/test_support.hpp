#pragma once

#include <filesystem>
#include <string>

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    path_ = std::filesystem::temp_directory_path() /
            ("mls_obstacle_test_" + std::to_string(++counter_));
    std::filesystem::create_directories(path_);
  }
  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }
  const std::filesystem::path& path() const { return path_; }

 private:
  inline static std::size_t counter_{0};
  std::filesystem::path path_;
};

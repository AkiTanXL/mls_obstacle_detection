#pragma once

#include "mls/obstacle_detection/types.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace mls::obstacle_detection {

class FrameSource {
 public:
  virtual ~FrameSource() = default;
  virtual bool next(Frame& frame) = 0;
  virtual void reset() = 0;
  [[nodiscard]] virtual std::size_t size() const = 0;
};

class PcdFrameSource final : public FrameSource {
 public:
  explicit PcdFrameSource(const std::filesystem::path& input);
  bool next(Frame& frame) override;
  void reset() override;
  [[nodiscard]] std::size_t size() const override;
  [[nodiscard]] const std::vector<std::filesystem::path>& paths() const;

 private:
  std::vector<std::filesystem::path> paths_;
  std::size_t cursor_{0};
};

class BinFrameSource final : public FrameSource {
 public:
  explicit BinFrameSource(const std::filesystem::path& input);
  bool next(Frame& frame) override;
  void reset() override;
  [[nodiscard]] std::size_t size() const override;
  [[nodiscard]] const std::vector<std::filesystem::path>& paths() const;

 private:
  std::vector<std::filesystem::path> paths_;
  std::size_t cursor_{0};
};

[[nodiscard]] std::unique_ptr<FrameSource> makeFrameSource(
    const std::filesystem::path& input);

}  // namespace mls::obstacle_detection

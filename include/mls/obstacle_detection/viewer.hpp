#pragma once

#include "mls/obstacle_detection/types.hpp"

#include <memory>

namespace mls::obstacle_detection {

enum class ViewerCommand { none, toggle_pause, step, restart, reset_view, quit };

class DetectionViewer {
 public:
  DetectionViewer();
  ~DetectionViewer();
  DetectionViewer(const DetectionViewer&) = delete;
  DetectionViewer& operator=(const DetectionViewer&) = delete;
  void show(const DetectionResult& result);
  void resetCamera();
  ViewerCommand spinOnce(int milliseconds);
  [[nodiscard]] bool stopped() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mls::obstacle_detection

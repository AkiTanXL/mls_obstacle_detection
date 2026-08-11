#include "mls/obstacle_detection/viewer.hpp"

#include <stdexcept>

namespace mls::obstacle_detection {

struct DetectionViewer::Impl {};
DetectionViewer::DetectionViewer() { throw std::runtime_error("viewer support was disabled at build time"); }
DetectionViewer::~DetectionViewer() = default;
void DetectionViewer::show(const DetectionResult&) {}
void DetectionViewer::resetCamera() {}
ViewerCommand DetectionViewer::spinOnce(int) { return ViewerCommand::quit; }
bool DetectionViewer::stopped() const { return true; }

}  // namespace mls::obstacle_detection

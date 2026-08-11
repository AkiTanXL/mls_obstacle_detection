#include "mls/obstacle_detection/viewer.hpp"
#include "mls/obstacle_detection/camera.hpp"

#include <pcl/visualization/pcl_visualizer.h>

#include <atomic>
#include <optional>
#include <string>

namespace mls::obstacle_detection {

struct DetectionViewer::Impl {
  pcl::visualization::PCLVisualizer::Ptr viewer{new pcl::visualization::PCLVisualizer("MLS obstacle detector")};
  std::atomic<ViewerCommand> command{ViewerCommand::none};
  std::optional<CameraPose> latest_camera;
  bool camera_initialized{false};

  void applyCamera() {
    if (!latest_camera) return;
    const auto& camera = *latest_camera;
    viewer->setCameraFieldOfView(camera.vertical_fov_rad);
    viewer->setCameraPosition(
        camera.position[0], camera.position[1], camera.position[2],
        camera.focal_point[0], camera.focal_point[1], camera.focal_point[2],
        camera.up[0], camera.up[1], camera.up[2]);
    viewer->setCameraClipDistances(camera.near_clip, camera.far_clip);
    camera_initialized = true;
  }

  Impl() {
    viewer->setSize(1280, 720);
    viewer->setBackgroundColor(0.82, 0.82, 0.82);
    viewer->addCoordinateSystem(1.0);
    viewer->registerKeyboardCallback([this](const pcl::visualization::KeyboardEvent& event) {
      if (!event.keyDown()) return;
      const std::string key = event.getKeySym();
      if (key == "space") command.store(ViewerCommand::toggle_pause);
      else if (key == "Right" || key == "n" || key == "N") command.store(ViewerCommand::step);
      else if (key == "r" || key == "R") command.store(ViewerCommand::restart);
      else if (key == "f" || key == "F") command.store(ViewerCommand::reset_view);
      else if (key == "q" || key == "Q" || key == "Escape") command.store(ViewerCommand::quit);
    });
  }
};

DetectionViewer::DetectionViewer() : impl_(std::make_unique<Impl>()) {}
DetectionViewer::~DetectionViewer() = default;

void DetectionViewer::show(const DetectionResult& result) {
  impl_->latest_camera = computeForwardCamera(result);
  impl_->viewer->removePointCloud("labeled_points");
  impl_->viewer->removeAllShapes();
  auto display = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>);
  display->width = result.labeled_cloud->width;
  display->height = result.labeled_cloud->height;
  display->is_dense = result.labeled_cloud->is_dense;
  display->resize(result.labeled_cloud->size());
  for (std::size_t i = 0; i < result.labeled_cloud->size(); ++i) {
    const auto& source = result.labeled_cloud->points[i];
    auto& target = display->points[i];
    target.x = source.x;
    target.y = source.y;
    target.z = source.z;
    target.rgba = source.rgba;
  }
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> colors(display);
  impl_->viewer->addPointCloud(display, colors, "labeled_points");
  impl_->viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.0, "labeled_points");
  for (const auto& obstacle : result.obstacles) {
    const std::string id = "obstacle_" + std::to_string(obstacle.instance_id);
    impl_->viewer->addCube(
        obstacle.aabb.min[0], obstacle.aabb.max[0], obstacle.aabb.min[1], obstacle.aabb.max[1],
        obstacle.aabb.min[2], obstacle.aabb.max[2], obstacle.color[0] / 255.0,
        obstacle.color[1] / 255.0, obstacle.color[2] / 255.0, id);
    impl_->viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                                                pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, id);
    impl_->viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2.0, id);
  }
  impl_->viewer->addText("frame: " + result.frame_id + "  obstacles: " +
                             std::to_string(result.obstacles.size()) +
                             "  |  F: forward fit",
                         10, 10, 14, 0.1, 0.1, 0.1, "status");
  if (!impl_->camera_initialized) impl_->applyCamera();
}

void DetectionViewer::resetCamera() { impl_->applyCamera(); }

ViewerCommand DetectionViewer::spinOnce(int milliseconds) {
  impl_->viewer->spinOnce(milliseconds, true);
  return impl_->command.exchange(ViewerCommand::none);
}

bool DetectionViewer::stopped() const { return impl_->viewer->wasStopped(); }

}  // namespace mls::obstacle_detection

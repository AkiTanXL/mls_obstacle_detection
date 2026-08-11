#include "mls/obstacle_detection/config.hpp"
#include "mls/obstacle_detection/frame_source.hpp"
#include "mls/obstacle_detection/pipeline.hpp"
#include "mls/obstacle_detection/serialization.hpp"
#include "mls/obstacle_detection/viewer.hpp"

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace od = mls::obstacle_detection;
using Clock = std::chrono::steady_clock;

namespace {

struct RunState {
  std::size_t succeeded{0};
  std::size_t skipped{0};
  std::vector<double> algorithm_times;
  nlohmann::json errors = nlohmann::json::array();
  std::set<std::string> saved_frames;
};

double percentile95(std::vector<double> values) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const std::size_t index = static_cast<std::size_t>(std::ceil(0.95 * values.size())) - 1;
  return values[index];
}

std::string summaryJson(const RunState& state, std::size_t total) {
  const double sum = std::accumulate(state.algorithm_times.begin(), state.algorithm_times.end(), 0.0);
  const double average = state.algorithm_times.empty() ? 0.0 : sum / state.algorithm_times.size();
  nlohmann::json summary{
      {"schema_version", 1}, {"frames_discovered", total}, {"frames_succeeded", state.succeeded},
      {"frames_skipped", state.skipped}, {"algorithm_ms", {{"average", average},
      {"p95", percentile95(state.algorithm_times)}, {"soft_target_average_ms", 100.0}}},
      {"errors", state.errors}};
  return summary.dump(2) + "\n";
}

bool handleFrame(od::FrameSource& source, const od::ObstacleDetectionPipeline& pipeline,
                 const od::PipelineConfig& config, const std::filesystem::path& output,
                 bool fail_fast, RunState& state, od::DetectionResult& displayed) {
  od::Frame frame;
  const auto read_begin = Clock::now();
  try {
    if (!source.next(frame)) return false;
  } catch (const std::exception& error) {
    ++state.skipped;
    state.errors.push_back({{"stage", "read"}, {"reason", error.what()}});
    std::cerr << "[skip] " << error.what() << '\n';
    if (fail_fast) throw;
    return true;
  }
  const double read_ms = std::chrono::duration<double, std::milli>(Clock::now() - read_begin).count();
  try {
    displayed = pipeline.process(frame);
    displayed.timings.read_ms = read_ms;
  } catch (const std::exception& error) {
    ++state.skipped;
    state.errors.push_back({{"source", frame.source_path.string()}, {"stage", "algorithm"}, {"reason", error.what()}});
    std::cerr << "[skip] " << frame.source_path << ": " << error.what() << '\n';
    if (fail_fast) throw;
    return true;
  }

  if (state.saved_frames.insert(frame.id).second) {
    od::writeFrameResultAtomic(od::outputPathsFor(output, frame), frame, displayed,
                               config.save.binary_compressed);
    ++state.succeeded;
    state.algorithm_times.push_back(displayed.timings.algorithm_ms);
    const double retained = displayed.counts.valid == 0 ? 0.0 :
        100.0 * displayed.counts.roi / displayed.counts.valid;
    const double ground = displayed.counts.voxel == 0 ? 0.0 :
        100.0 * displayed.counts.ground_voxel / displayed.counts.voxel;
    std::cout << "[ok] " << frame.source_path.filename() << " points=" << displayed.counts.input
              << " crop=" << retained << "% ground=" << ground << "% obstacles="
              << displayed.obstacles.size() << " algorithm=" << displayed.timings.algorithm_ms
              << "ms save=" << displayed.timings.save_ms << "ms\n";
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"PCL-based vehicle LiDAR obstacle detector"};
  std::filesystem::path config_path;
  std::filesystem::path input_path;
  std::filesystem::path output_path;
  bool force_viewer = false;
  bool no_viewer = false;
  bool loop = false;
  bool overwrite = false;
  bool fail_fast = false;
  std::optional<double> fps;
  app.add_option("--config", config_path, "Pipeline YAML configuration")->required()->check(CLI::ExistingFile);
  app.add_option("--input", input_path, "PCD or KITTI BIN file/directory")
      ->required()->check(CLI::ExistingPath);
  app.add_option("--output", output_path, "Output directory")->required();
  auto* viewer_option = app.add_flag("--viewer", force_viewer, "Enable visualization (default)");
  auto* no_viewer_option = app.add_flag("--no-viewer", no_viewer, "Disable visualization");
  viewer_option->excludes(no_viewer_option);
  app.add_option("--fps", fps, "Viewer playback rate")->check(CLI::PositiveNumber);
  app.add_flag("--loop", loop, "Replay from the first frame after the last");
  app.add_flag("--overwrite", overwrite, "Allow writing into a non-empty output directory");
  app.add_flag("--fail-fast", fail_fast, "Stop on the first bad frame");
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& error) {
    return app.exit(error);
  }

  try {
    od::PipelineConfig config = od::loadConfig(config_path);
    (void)force_viewer;
    if (fps) config.playback.fps = *fps;
    if (loop) config.playback.loop = true;
    od::validateConfig(config);
    auto source = od::makeFrameSource(input_path);
    od::prepareOutputDirectory(output_path, overwrite);
    od::writeTextAtomic(output_path / "resolved_config.yaml", od::configToYaml(config));
    od::ObstacleDetectionPipeline pipeline(config);
    RunState state;
    const bool use_viewer = !no_viewer;
    if (!use_viewer) {
      od::DetectionResult unused;
      while (handleFrame(*source, pipeline, config, output_path, fail_fast, state, unused)) {}
    } else {
      auto viewer = std::make_unique<od::DetectionViewer>();
      bool paused = false;
      bool advance = true;
      auto deadline = Clock::now();
      od::DetectionResult displayed;
      while (!viewer->stopped()) {
        const auto command = viewer->spinOnce(10);
        if (command == od::ViewerCommand::quit) break;
        if (command == od::ViewerCommand::toggle_pause) paused = !paused;
        if (command == od::ViewerCommand::step && paused) advance = true;
        if (command == od::ViewerCommand::restart) {
          source->reset();
          advance = true;
          deadline = Clock::now();
        }
        if (command == od::ViewerCommand::reset_view) viewer->resetCamera();
        if (!paused && Clock::now() >= deadline) advance = true;
        if (!advance) continue;
        if (!handleFrame(*source, pipeline, config, output_path, fail_fast, state, displayed)) {
          if (config.playback.loop) {
            source->reset();
            continue;
          }
          break;
        }
        if (!displayed.frame_id.empty()) viewer->show(displayed);
        advance = false;
        deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(1.0 / config.playback.fps));
      }
    }
    od::writeTextAtomic(output_path / "run_summary.json", summaryJson(state, source->size()));
    return state.skipped == 0 ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << '\n';
    return 1;
  }
}

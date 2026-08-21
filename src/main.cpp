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
#include <utility>
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

enum class ProcessStatus { end, skipped, success };

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

od::FrameSaveCompletion frameSaveCompletion(const od::DetectionResult& result) {
  return {result.source_path, result.counts, result.obstacles.size(),
          result.filtered_clusters.size(), result.timings.algorithm_ms, result.timings.save_ms};
}

void recordSavedFrame(const od::FrameSaveCompletion& completion, RunState& state) {
  ++state.succeeded;
  state.algorithm_times.push_back(completion.algorithm_ms);
  const double retained = completion.counts.valid == 0 ? 0.0 :
      100.0 * completion.counts.roi / completion.counts.valid;
  const double ground = completion.counts.voxel == 0 ? 0.0 :
      100.0 * completion.counts.ground_voxel / completion.counts.voxel;
  std::cout << "[ok] " << completion.source_path.filename() << " points="
            << completion.counts.input << " crop=" << retained << "% ground=" << ground
            << "% obstacles=" << completion.obstacle_count << " filtered="
            << completion.filtered_cluster_count << " algorithm=" << completion.algorithm_ms
            << "ms save=" << completion.save_ms << "ms\n";
}

ProcessStatus processNextFrame(od::FrameSource& source,
                               const od::ObstacleDetectionPipeline& pipeline,
                               bool fail_fast, RunState& state, od::Frame& frame,
                               od::DetectionResult& result) {
  frame = od::Frame{};
  result = od::DetectionResult{};
  const auto read_begin = Clock::now();
  try {
    if (!source.next(frame)) return ProcessStatus::end;
  } catch (const std::exception& error) {
    ++state.skipped;
    state.errors.push_back({{"stage", "read"}, {"reason", error.what()}});
    std::cerr << "[skip] " << error.what() << '\n';
    if (fail_fast) throw;
    return ProcessStatus::skipped;
  }
  const double read_ms = std::chrono::duration<double, std::milli>(Clock::now() - read_begin).count();
  try {
    result = pipeline.process(frame);
    result.timings.read_ms = read_ms;
  } catch (const std::exception& error) {
    ++state.skipped;
    state.errors.push_back({{"source", frame.source_path.string()}, {"stage", "algorithm"}, {"reason", error.what()}});
    std::cerr << "[skip] " << frame.source_path << ": " << error.what() << '\n';
    if (fail_fast) throw;
    return ProcessStatus::skipped;
  }
  return ProcessStatus::success;
}

void collectCompletions(od::AsyncFrameWriter& writer, RunState& state) {
  for (const auto& completion : writer.takeCompletions()) {
    recordSavedFrame(completion, state);
  }
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
      od::Frame frame;
      od::DetectionResult result;
      while (true) {
        const auto status = processNextFrame(*source, pipeline, fail_fast, state, frame, result);
        if (status == ProcessStatus::end) break;
        if (status == ProcessStatus::skipped) continue;
        if (!state.saved_frames.insert(frame.id).second) continue;
        od::writeFrameResultAtomic(od::outputPathsFor(output_path, frame), result,
                                   config.save.binary_compressed);
        recordSavedFrame(frameSaveCompletion(result), state);
      }
    } else {
      auto viewer = std::make_unique<od::DetectionViewer>();
      od::AsyncFrameWriter writer(config.save.binary_compressed);
      bool paused = false;
      bool advance = true;
      auto deadline = Clock::now();
      od::Frame frame;
      od::DetectionResult displayed;
      while (!viewer->stopped()) {
        writer.rethrowIfFailed();
        collectCompletions(writer, state);
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
        const auto status = processNextFrame(*source, pipeline, fail_fast, state, frame, displayed);
        if (status == ProcessStatus::end) {
          if (config.playback.loop) {
            source->reset();
            continue;
          }
          break;
        }
        if (status == ProcessStatus::success) {
          viewer->show(displayed);
          if (state.saved_frames.insert(frame.id).second) {
            writer.enqueue(od::outputPathsFor(output_path, frame), std::move(displayed));
          }
        }
        advance = false;
        deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(1.0 / config.playback.fps));
      }
      writer.finish();
      collectCompletions(writer, state);
    }
    od::writeTextAtomic(output_path / "run_summary.json", summaryJson(state, source->size()));
    return state.skipped == 0 ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << '\n';
    return 1;
  }
}

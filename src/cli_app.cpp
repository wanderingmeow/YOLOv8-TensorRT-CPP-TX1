/*******************************************************************************
 * A simple CLI App for inferring YoloV8 models with TensorRT.
 * Copyright (c) 2024 WanderingMeow
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 ******************************************************************************/

#include <cli/cli.h>
#include <cli/clilocalsession.h>
#include <cli/colorprofile.h>

#include "cmd_line_util.h"
#include "yolov8.h"

#include <atomic>
#include <csignal>
#include <stdexcept>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

#ifdef CLI_USE_STANDALONEASIO_SCHEDULER
#include <cli/standaloneasioremotecli.h>
#include <cli/standaloneasioscheduler.h>
namespace cli {
using MainScheduler = StandaloneAsioScheduler;
using CliTelnetServer = StandaloneAsioCliTelnetServer;
} // namespace cli
#elif defined(CLI_USE_BOOSTASIO_SCHEDULER)
#include <cli/boostasioremotecli.h>
#include <cli/boostasioscheduler.h>
namespace cli {
using MainScheduler = BoostAsioScheduler;
using CliTelnetServer = BoostAsioCliTelnetServer;
} // namespace cli
#else
#error either CLI_USE_STANDALONEASIO_SCHEDULER or CLI_USE_BOOSTASIO_SCHEDULER must be defined
#endif

void getScreenResolution(int &width, int &height) {
    width = 1280;
    height = 720;
}

enum class InferenceState { Idle, Loaded, Inferring };

class InferenceServer {
public:
    InferenceServer(YoloV8Config config) : config(config) {
        int width, height;
        getScreenResolution(width, height);
        windowWidth = static_cast<int>(0.9 * width);
        windowHeight = static_cast<int>(0.9 * height);
    }

    ~InferenceServer() {}

    int loadModel(const std::string &modelPath) {
        if (modelPath_ == modelPath && state != InferenceState::Idle) {
            std::cout << "Model already loaded.\n";
            return 0;
        }

        state = InferenceState::Idle;

        std::string modelExt =
            modelPath.substr(modelPath.find_last_of(".") + 1);
        std::string onnxModelPath = (modelExt == "onnx") ? modelPath : "";
        std::string trtModelPath =
            modelPath.find(".engine") != std::string::npos ? modelPath : "";

        if (onnxModelPath.empty() && trtModelPath.empty()) {
            std::cerr << "Unsupported model file format.\n";
            return -1;
        }

        try {
            engine =
                std::make_unique<YoloV8>(onnxModelPath, trtModelPath, config);
            modelPath_ = modelPath;
            warmUpModel();
            state = InferenceState::Loaded;
            return 0;
        } catch (const std::exception &e) {
            std::cerr << "Failed to load model: " << e.what() << "\n";
            return -1;
        }
    }

    void startInference(const std::string &filePath,
                        std::function<void()> doneCallback) {
        if (modelPath_.empty()) {
            std::cerr << "No model loaded. Please load a model first.\n";
            return;
        }

        shouldStopInference = false;
        state = InferenceState::Inferring;

        try {
            handleInputFile(filePath);
            doneCallback();
        } catch (const std::exception &e) {
            std::cerr << "Failed to process file '" << filePath
                      << "': " << e.what() << "\n";
        }

        state = InferenceState::Loaded;
    }

    void stopInference() { shouldStopInference = true; }

    std::string getStats() const {
        std::ostringstream stats;
        if (modelPath_.empty()) {
            stats << "No model loaded.\n";
        } else {
            stats << "Model Path: '" << modelPath_ << "'\n";
        }
        return stats.str();
    }

private:
    void warmUpModel() {
        std::cout << "Warming up the model...\n";
        cv::Mat fakeImage = cv::Mat::zeros(1280, 720, CV_8UC3);
        for (int i = 0; i < 10; ++i) {
            engine->detectObjects(fakeImage);
        }
    }

    void handleInputFile(const std::string &filePath) {
        cv::Mat frame = cv::imread(filePath);
        cv::VideoCapture cap;

        if (!frame.empty()) {
            processImage(frame);
        } else if (cap.open(filePath)) {
            processVideo(cap);
        } else {
            throw std::runtime_error("Unable to open '" + filePath +
                                     "' as an image or a video.");
        }
    }

    void processVideo(cv::VideoCapture &cap) {
        unsigned int frames = 0;
        int64_t start_tick = cv::getTickCount();

        cv::Mat img;

        while (cap.read(img) && !shouldStopInference) {
            if (img.empty()) {
                throw std::runtime_error(
                    "Unable to decode image from video stream.");
            }

            img = runInference(img);

            frames++;
            double timeElapsed = (cv::getTickCount() - start_tick) * 1000. /
                                 (frames * cv::getTickFrequency());
            std::ostringstream strStream;
            strStream << std::fixed << std::setprecision(4) << timeElapsed;

            std::cout << "Frame time: " << strStream.str() << " ms\n";

            cv::imshow(windowName, img);
            int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {
                break;
            }
        }

        cv::destroyAllWindows();
    }

    void processImage(cv::Mat &img) {
        img = runInference(img);

        cv::imshow(windowName, img);
        cv::waitKey(-1);
        cv::destroyAllWindows();
    }

    cv::Mat runInference(cv::Mat &img, bool resizeToFitWindow = true) {
        const auto objects = engine->detectObjects(img);
        engine->drawObjectLabels(img, objects);
        if (resizeToFitWindow) {
            double aspectRatio = static_cast<double>(img.cols) / img.rows;
            int width, height;
            if (windowWidth / aspectRatio <= windowHeight) {
                width = windowWidth;
                height = static_cast<int>(windowWidth / aspectRatio);
            } else {
                height = windowHeight;
                width = static_cast<int>(windowHeight * aspectRatio);
            }
            cv::Mat resized;
            cv::resize(img, resized, cv::Size(width, height));
            return resized;
        }
        return img;
    }

    const std::string windowName = "YoloV8 Inference Result";

    std::unique_ptr<YoloV8> engine;
    YoloV8Config config;

    int windowWidth, windowHeight;

    std::atomic<InferenceState> state;
    std::string modelPath_;
    std::atomic<bool> shouldStopInference{false};
};

static std::optional<std::function<void(int)>> ctrlCHandler;

void signalHandlerFunc(int signum) {
    if (signum == SIGINT && ctrlCHandler) {
        (*ctrlCHandler)(signum);
    }
}

int main(int argc, char *argv[]) {
    YoloV8Config config;
    std::string onnxModelPath;
    std::string trtModelPath;

    // Parse the command line arguments
    if (!parseArgumentsInteractive(argc, argv, config, onnxModelPath,
                                   trtModelPath)) {
        return -1;
    }

    std::unique_ptr<InferenceServer> server =
        std::make_unique<InferenceServer>(config);

    int res = -1;
    if (!onnxModelPath.empty()) {
        res = server->loadModel(onnxModelPath);
    }
    if (res && !trtModelPath.empty()) {
        res = server->loadModel(trtModelPath);
    }

    try {
        auto rootMenu = std::make_unique<cli::Menu>("yolov8");

        rootMenu->Insert(
            "load", {"model_path"},
            [&server](std::ostream &out, const std::string &modelPath) {
                server->loadModel(modelPath);
                out << "Model loaded and warmed up.\n";
            },
            "Load and warm up the specified YOLOv8 model.");

        rootMenu->Insert(
            "infer", {"file_path"},
            [&server](std::ostream &out, const std::string &filePath) {
                server->startInference(filePath, [&]() {});
            },
            "Start inference on the specified image or video.");

        rootMenu->Insert(
            "info", {},
            [&server](std::ostream &out) { out << server->getStats(); },
            "Print information about the currently loaded model.");

        cli::Cli cli(std::move(rootMenu));
        cli::SetColor();
        cli.ExitAction([](auto &out) {});
        cli.StdExceptionHandler([](std::ostream &out, const std::string &cmd,
                                   const std::exception &e) {
            out << "Exception caught in CLI handler: " << e.what()
                << " while handling command: " << cmd << ".\n";
        });
        cli.WrongCommandHandler([](std::ostream &out, const std::string &cmd) {
            out << "Unknown command or incorrect parameters: " << cmd << ".\n";
        });

        cli::MainScheduler scheduler;
        cli::CliLocalTerminalSession localSession(cli, scheduler, std::cout,
                                                  200);
        localSession.ExitAction([&scheduler](auto &out) {
            out << "Exiting...\n";
            scheduler.Stop();
        });

        auto signalHandler = [&](int signum) {
            if (signum == SIGINT) {
                std::cout << "\n";
                server->stopInference();
                localSession.Prompt();
            }
        };

        ctrlCHandler = signalHandler;

        std::signal(SIGINT, signalHandlerFunc);

        scheduler.Run();
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Exception caught in main: " << e.what() << '\n';
    } catch (...) {
        std::cerr << "Unknown exception caught in main.\n";
    }

    return -1;
}
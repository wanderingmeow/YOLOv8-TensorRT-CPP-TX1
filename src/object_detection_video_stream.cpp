#include "cmd_line_util.h"
#include "yolov8.h"
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui.hpp>
#include <sstream>
#include <string>

// Runs object detection on video stream then displays annotated results.
int main(int argc, char *argv[]) {
    YoloV8Config config;
    std::string onnxModelPath;
    std::string trtModelPath;
    std::string inputVideo;

    // Parse the command line arguments
    if (!parseArgumentsVideo(argc, argv, config, onnxModelPath, trtModelPath,
                             inputVideo)) {
        return -1;
    }

    // Create the YoloV8 engine
    YoloV8 yoloV8(onnxModelPath, trtModelPath, config);

    // Initialize the video stream
    cv::VideoCapture cap;

    // Open video capture
    try {
        cap.open(std::stoi(inputVideo));
    } catch (const std::exception &e) {
        cap.open(inputVideo);
    }

    // Try to use HD resolution (or closest resolution)
    auto resW = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    auto resH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "Original video resolution: (" << resW << "x" << resH << ")"
              << std::endl;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    resW = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    resH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "New video resolution: (" << resW << "x" << resH << ")"
              << std::endl;

    if (!cap.isOpened())
        throw std::runtime_error("Unable to open video capture with input '" +
                                 inputVideo + "'");

    // Warm-up model
    std::cout << "Warming up...\n";
    cv::Mat fakeImage = cv::Mat::zeros(resH, resW, CV_8UC3);
    for (int i = 0; i < 10; i++) {
        yoloV8.detectObjects(fakeImage);
    }

    unsigned int frames = 0;
    int64_t start_time = cv::getTickCount();

    while (true) {
        // Grab frame
        cv::Mat img;
        cap >> img;

        if (img.empty())
            throw std::runtime_error(
                "Unable to decode image from video stream.");

        // Run inference
        const auto objects = yoloV8.detectObjects(img);

        // Draw the bounding boxes on the image
        yoloV8.drawObjectLabels(img, objects);

        // Calulate FPS
        frames++;
        double timeElapsed =
            (cv::getTickCount() - start_time) / cv::getTickFrequency();
        double fps = frames / timeElapsed;

        // Print FPS text on the image
        std::ostringstream strStream;
        strStream << std::fixed << std::setprecision(2) << fps;
        std::string fpsText = "FPS: " + strStream.str();
        cv::putText(img, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                    1, cv::Scalar(0, 255, 0), 2);

        // Display the results
        cv::imshow("Object Detection", img);
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q')
            break;
    }

    cv::destroyWindow("Object Detection");

    return 0;
}
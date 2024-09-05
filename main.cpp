#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include "BYTETracker.h"  // 假设存在一个 ByteTrack 类用于对象跟踪
#include <random>

using namespace cv;
using namespace std;

struct Detection
{
    int class_id{ 0 };
//    std::string className{};
    float confidence{ 0.0 };
//    cv::Scalar color{};
    cv::Rect box{};
};
//std::vector<Detection> detections{};
std::vector<std::string> m_classes{ "person", "bicycle", "car", "motorcycle",
                                    "airplane", "bus", "train", "truck", "boat", "traffic light",
                                    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
                                    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
                                    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
                                    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                                    "baseball bat", "baseball glove", "skateboard", "surfboard",
                                    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
                                    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                                    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                    "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                                    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                                    "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                    "scissors", "teddy bear", "hair drier", "toothbrush" };
// 函数声明
std::vector<std::tuple<int, float, Rect>> detectObjects(dnn::Net& net, const Mat& frame, float confThreshold);


// 检测函数
vector<tuple<int, float, Rect>> detectObjects(dnn::Net& net, const Mat& frame, float confThreshold) {
    Mat blob;
    dnn::blobFromImage(frame, blob, 1 / 255.0, Size(640, 640), Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    vector<Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    int rows = outs[0].size[1];
    int dimensions = outs[0].size[2];

    bool yolov8 = false;
    if (dimensions > rows) {
        yolov8 = true;
        rows = outs[0].size[2];
        dimensions = outs[0].size[1];

        outs[0] = outs[0].reshape(1, dimensions);
        cv::transpose(outs[0], outs[0]);
    }

    float* data = (float*)outs[0].data;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    int frameWidth = frame.cols;
    int frameHeight = frame.rows;
    float x_factor = frameWidth / 640.0;
    float y_factor = frameHeight / 640.0;

    for (int i = 0; i < rows; ++i) {
        if (yolov8) {
            float* classes_scores = data + 4;
            cv::Mat scores(1, m_classes.size(), CV_32FC1, classes_scores);

            cv::Point class_id;
            double maxClassScore;
            minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

            if (maxClassScore > 0.5) {
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = static_cast<int>((x - 0.5 * w) * x_factor);
                int top = static_cast<int>((y - 0.5 * h) * y_factor);
                int width = static_cast<int>(w * x_factor);
                int height = static_cast<int>(h * y_factor);

                boxes.push_back(cv::Rect(left, top, width, height));
            }
        } else {
            float confidence = data[4];

            if (confidence >= 0.4) {
                float* classes_scores = data + 5;
                cv::Mat scores(1, m_classes.size(), CV_32FC1, classes_scores);

                cv::Point class_id;
                double max_class_score;
                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

                if (max_class_score > 0.5) {
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);

                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];

                    int left = static_cast<int>((x - 0.5 * w) * x_factor);
                    int top = static_cast<int>((y - 0.5 * h) * y_factor);
                    int width = static_cast<int>(w * x_factor);
                    int height = static_cast<int>(h * y_factor);

                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }

        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.7, nms_result);

    vector<tuple<int, float, Rect>> detections;
    for (unsigned long i = 0; i < nms_result.size(); ++i) {
        int idx = nms_result[i];

        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];

        detections.push_back(make_tuple(result.class_id, result.confidence, result.box));
    }
    return detections;
}


class line_point {
public:
    int x, y;
    line_point(int x, int y) : x(x), y(y) {}
};

class DetectionsTrack {
public:
    vector<tuple<Rect, float, int, int>> detections; // (xyxy, confidence, class_id, tracker_id)

    void add(const Rect& xyxy, float confidence, int class_id, int tracker_id) {
        detections.push_back(make_tuple(xyxy, confidence, class_id, tracker_id));
    }
};

bool isInLine(const line_point& pt1, const line_point& pt2, const line_point& pt) {
    int x1 = pt1.x, y1 = pt1.y;
    int x2 = pt2.x, y2 = pt2.y;
    int x = pt.x, y = pt.y;
    return ((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) >= 0;
}

// Trigger function to count crossings
pair<int, int> trigger(const DetectionsTrack& detections_track, const line_point& pt1, const line_point& pt2,
                       map<int, map<string, int>>& prev_tracker_state, map<int, map<string, int>>& tracker_state,
                       set<int>& crossing_ids, int in_count, int out_count) {
    for (const auto& detection : detections_track.detections) {
        Rect xyxy;
        float confidence;
        int class_id;
        int tracker_id;

        // 使用 std::tie 解包 tuple
        tie(xyxy, confidence, class_id, tracker_id) = detection;

        int x1 = xyxy.x;
        int y1 = xyxy.y;
        int x2 = xyxy.x + xyxy.width;
        int y2 = xyxy.y + xyxy.height;
        line_point center((x1 + x2) / 2, (y1 + y2) / 2);
        bool tracker_state_new = isInLine(pt1, pt2, center);

        if (tracker_state.find(tracker_id) == tracker_state.end() || tracker_state[tracker_id].empty()) {
            tracker_state[tracker_id] = {{"state", tracker_state_new}, {"direction", -1}};
            if (prev_tracker_state.find(tracker_id) != prev_tracker_state.end() && !prev_tracker_state[tracker_id].empty()) {
                tracker_state[tracker_id]["direction"] = prev_tracker_state[tracker_id]["direction"];
            }
        } else if (tracker_state[tracker_id]["state"] == tracker_state_new) {
            continue;
        } else {
            if (tracker_state[tracker_id]["state"] && !tracker_state_new) {
                if (tracker_state[tracker_id]["direction"] != 1) {
                    in_count++;
                }
                tracker_state[tracker_id]["direction"] = 1;
            } else if (!tracker_state[tracker_id]["state"] && tracker_state_new) {
                if (tracker_state[tracker_id]["direction"] != 0) {
                    out_count++;
                }
                tracker_state[tracker_id]["direction"] = 0;
            }
            tracker_state[tracker_id]["state"] = tracker_state_new;
        }
    }

    // 更新丢失的检测
    set<int> current_ids;
    for (const auto& detection : detections_track.detections) {
        Rect xyxy;
        float confidence;
        int class_id, tracker_id;
        std::tie(xyxy, confidence, class_id, tracker_id) = detection;
        current_ids.insert(tracker_id);
    }

    for (auto it = tracker_state.begin(); it != tracker_state.end();) {
        if (current_ids.find(it->first) == current_ids.end()) {
            prev_tracker_state[it->first] = it->second;
            it = tracker_state.erase(it);
        } else {
            ++it;
        }
    }

    return {in_count, out_count};
}





int main() {
    // 1. 加载 YOLOv8 ONNX 模型
    string modelPath = "/home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/yolov8s.onnx"; // 请根据实际路径修改
    dnn::Net net = dnn::readNetFromONNX(modelPath);
    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);

    // 2. 打开视频文件或相机
    string VIDEO_PATH = "/home/ai/data/zhangjunming/yolov8_bytetrack/ultralytics/video/test_person.mp4";
    VideoCapture cap(VIDEO_PATH); // 打开视频文件
    if (!cap.isOpened()) {
        cerr << "Error opening video file or camera" << endl;
        return -1;
    }

    // 3. 初始化 VideoWriter 对象
    string outputVideoPath = "/home/ai/data/zhangjunming/yolov8_bytetrack/ultralytics/video/output_test_person.mp4";
//    string outputVideoPath = "/home/ai/data/zhangjunming/yolov8_bytetrack/ultralytics/video/output_test_traffic2.jpg";

//    int codec = VideoWriter::fourcc('M', 'J', 'P', 'G'); // 选择一个编解码器
    int codec = VideoWriter::fourcc('H', '2', '6', '4'); // 使用 H.264 编码器

    double fps = cap.get(CAP_PROP_FPS); // 获取输入视频的帧率
    Size frameSize = Size((int)cap.get(CAP_PROP_FRAME_WIDTH), (int)cap.get(CAP_PROP_FRAME_HEIGHT));
    VideoWriter writer(outputVideoPath, codec, fps, frameSize, true);

    if (!writer.isOpened()) {
        cerr << "Error opening video writer" << endl;
        return -1;
    }

    // 4. 初始化 ByteTrack 跟踪器
    BYTETracker tracker(30,30);  // 需要提供 ByteTrack 实现


// 定义检测到的物体的ID
    map<int, map<string, int>> prev_tracker_state;
    map<int, map<string, int>> tracker_state;
    set<int> crossing_ids;
    int in_count = 0;
    int out_count = 0;


    Mat frame;
    int num_frames = 0;
    int detection_interval = 25; // 每 25 帧进行一次检测
    while (cap.read(frame)) {



        // 定义跨线的位置
        line_point pt1(0, frame.rows / 2);
        line_point pt2(frame.cols, frame.rows / 2);

        // 检查帧是否为空
        if (frame.empty()) {
            cerr << "空帧 detected!" << endl;
            break; // 如果帧为空，则退出循环
        }
        num_frames++;



        // 5. YOLOv8 对象检测
        float confThreshold = 0.4;  // 置信度阈值
        vector<tuple<int, float, Rect>> detections = detectObjects(net, frame, confThreshold);

        // 6. 提取检测框和创建 Object 对象
        vector<Object> objects;
        for (const auto &detection: detections) {
            int classId;
            float confidence;
            Rect box;
            tie(classId, confidence, box) = detection;

            Object obj;
            obj.rect = box;
            obj.label = classId;
            obj.prob = confidence;
            objects.push_back(obj);
        } //最后ByteTrack需要的东西就是 这个vector<Object> objects;  所以可以在aibox代码中做一个这种objects
        //最后就是需要拿到Aibox的detections这么搞：
//        for (const auto& detection : detections) {
//            Object obj;
//            obj.rect = detection.box;         // 使用Detection中的box
//            obj.label = detection.class_id;   // 使用Detection中的class_id
//            obj.prob = detection.confidence;  // 使用Detection中的confidence
//
//            objects.push_back(obj);           // 将转换后的Object对象存入objects中
//        }





        // 7. 使用 ByteTrack 进行跟踪  update会处理这些目标，并返回当前帧中所有目标的检测结果
        // trakcs是一个包含STrack对象的向量，每个STrack对象包含一个跟踪目标的信息
        vector<STrack> tracks = tracker.update(objects);

        // 将 ByteTrack 的跟踪结果转换为 DetectionsTrack 对象
        DetectionsTrack detections_track;
        for (const auto& track : tracks) {
            detections_track.add(
                    Rect(
                            static_cast<int>(track.tlwh[0]), // x
                            static_cast<int>(track.tlwh[1]), // y
                            static_cast<int>(track.tlwh[2]), // w
                            static_cast<int>(track.tlwh[3])  // h
                    ),
                    track.score,
                    track.track_id,
                    track.track_id
            );
        }


        //8 查看跨线的进出数量
        tie(in_count, out_count) = trigger(detections_track, pt1, pt2, prev_tracker_state, tracker_state, crossing_ids, in_count, out_count);
        std::cout<<"本帧的in_cout:"<<in_count << "out_count"<<out_count<<std::endl;

        // 更新 prev_tracker_state
        prev_tracker_state = tracker_state;

        // 清空 crossing_ids 用于下一帧
        crossing_ids.clear();


        // 9. 显示跟踪结果
        for (const auto &track: tracks) {
            // 从 tlwh 转换到 Rect
            const vector<float> &tlwh = track.tlwh;
            Rect bbox(
                    static_cast<int>(tlwh[0]),                     // x
                    static_cast<int>(tlwh[1]),                     // y
                    static_cast<int>(tlwh[2]),                     // w
                    static_cast<int>(tlwh[3])                      // h
            );

            rectangle(frame, bbox, Scalar(0, 255, 0), 2);
            string label = format("ID: %d Conf: %.2f", track.track_id, track.score);
            putText(frame, label, bbox.tl(), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 0, 0), 2);
        }


        //绘制跨线
        line(frame, Point(pt1.x, pt1.y), Point(pt2.x, pt2.y), Scalar(0, 0, 255), 2);
        // 显示计数信息
        string count_info = "In: " + to_string(in_count) + " Out: " + to_string(out_count);
        putText(frame, count_info, Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 2);

        // 10. 写入结果帧到输出视频
        writer.write(frame);
        if (waitKey(1) == 27) break;  // 按下 ESC 键退出
    }

    // 11. 释放资源
    cap.release();
    writer.release();
    destroyAllWindows();
    return 0;
}

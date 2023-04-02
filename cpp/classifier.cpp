#include <iostream>
#include <memory>
#include <chrono>
#include <map>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torchvision/vision.h>
#include <torchvision/ops/nms.h>

class ImageClassifier {

    protected:
    torch::jit::script::Module module;
    std::string modelPath;
    cv::VideoCapture cap;
    c10::Device device = c10::Device(c10::kCPU);
    std::vector<float> inference_times = {};
    std::vector<float> fpss = {};
    std::vector<int> preds = {};
    int ImgSize = 224;
    float average(std::vector<float> const& v) {
        if (v.empty()) {
            return 0;
        }
        auto const count = static_cast<float>(v.size());
        return std::reduce(v.begin(), v.end()) / count;
    }
    int mode(std::vector<int> const& vec) {

        std::map<int, int> freq;
        for (auto i : vec) { freq[i]++; }

        int mode = -1;
        int max_freq = 0;

        for (auto i: freq) {
            if (i.second > max_freq) {
                max_freq = i.second;
                mode = i.first;
            }
        }
        return mode;
    }
    
    public:
    void initModel() {
        try {
            module = torch::jit::load(this->modelPath);
            module.to(this->device);
            module.eval();

            if (module.is_training())
                module.eval();
        }
        catch (const c10::Error& e) {
            std::cerr << "Error loading the model\n";
            return;
        }
        std::cout << "Model loaded" << std::endl;
    }

    void testModel(int type) {
        if (type == 1) {
            try {
                torch::Tensor inputs = torch::rand({1, 3, ImgSize, ImgSize});
                torch::Tensor outputTensor = this->module.forward({inputs}).toTensor();
                std::cout << outputTensor.sizes() << std::endl;
            }
            catch (const c10::Error& e) {
                std::cerr << "Error test the model\n";
                std::cerr << e.msg() << std::endl;
                return;
            }
        } else if (type == 2) {
            try {
                torch::Tensor inputs = torch::rand({1, 3, ImgSize, ImgSize});
                auto output = this->module.forward({inputs}).toTupleRef();
                auto output1 = output.elements()[0].toTensor();
                auto output2 = output.elements()[1].toTensor();
                auto output3 = output.elements()[2].toTensor();
                // std::cout << output.sizes() << std::endl;
                std::cout << output1.sizes() << std::endl;
                std::cout << output2.sizes() << std::endl;
                std::cout << output3.sizes() << std::endl;
            }
            catch (const c10::Error& e) {
                std::cerr << "Error test the model\n";
                std::cerr << e.msg() << std::endl;
                return;
            }
        } else {
            try {
                torch::Tensor input1 = torch::rand({3, ImgSize, ImgSize});
                torch::Tensor input2 = torch::rand({3, ImgSize, ImgSize});
                std::vector<torch::Tensor> images = { input1 };
                auto output = this->module.forward({images}).toTuple().get()[0];
                auto losses = output.elements()[0].toGenericDict();
                auto results = output.elements()[1].toListRef();

                for (auto result : results) {
                    auto dict1 = result.toGenericDict();
                    auto boxes = dict1.at("boxes").toTensor();
                    auto labels = dict1.at("labels").toTensor();
                    auto scores = dict1.at("scores").toTensor();
                    std::cout << boxes << std::endl;
                    std::cout << labels << std::endl;
                    std::cout << scores << std::endl;
                }
            }
            catch (const c10::Error& e) {
                std::cerr << "Error test the model\n";
                std::cerr << e.msg() << std::endl;
                return;
            }
        }
    }

    private:
    int classifyImage(cv::Mat &image) {
          
        c10::InferenceMode guard;

        cv::Mat reImage;
        cv::Mat rgbImage;

        cv::resize(image, reImage, cv::Size(ImgSize,ImgSize));
        cv::cvtColor(reImage, rgbImage, cv::COLOR_BGR2RGB);

        torch::Tensor inputTensor = torch::from_blob(
            rgbImage.data, 
            {rgbImage.rows, rgbImage.cols, 3},
            torch::kByte
        ).to(torch::kFloat);
        inputTensor = inputTensor.div(255);
        inputTensor = inputTensor.permute({2,0,1});
        inputTensor.unsqueeze_(0);
        
        inputTensor.to(this->device);
        module.to(this->device);
        module.eval();

        std::vector<torch::jit::IValue> inputs = {inputTensor};
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        torch::Tensor outputTensor = module.forward(inputs).toTensor();

        auto end_time = std::chrono::high_resolution_clock::now();
        float time_ms = (float) std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        inference_times.push_back(time_ms);
        
        std::tuple result = outputTensor.max(1);

        auto max_value = std::get<0>(result);
        auto max_index = std::get<1>(result);

        return max_index.item().toInt();
    }

    public:
    void run() {
        
        int frames = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        if (!this->cap.isOpened()) {
            std::cout << "Error opening camera" << std::endl;
            return;
        }

        cv::namedWindow("Webcam", cv::WINDOW_NORMAL);

        while (true) {
            cv::Mat frame;
            this->cap.read(frame);

            if (frame.empty()) {
                std::cout << "Error reading frame" << std::endl;
                break;
            }

            int pred = this->classifyImage(frame);
            preds.push_back(pred);

            frames++;
            auto end_time = std::chrono::high_resolution_clock::now();
            auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            float fps = (float) frames * 1000.0 / elapsed_time;
            fpss.push_back(fps);

            cv::putText(frame, cv::format("FPS: %.2f", fps), cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, cv::format("PRED: %d", pred), cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

            cv::imshow("Webcam", frame);

            int key = cv::waitKey(1);

            if (key == 'q') {
                break;
            }
        }
    }

    void classifyImageFromPath(std::string path) {
        cv::Mat image = cv::imread(path);
        int pred = this->classifyImage(image);
        std::cout << "PRED: " << pred << std::endl;
    }

    void close() {
        this->cap.release();
        cv::destroyAllWindows();

        std::cout << "Average inference time: " << average(inference_times) << " ms" << std::endl;
        std::cout << "Average FPS: " << average(fpss) << std::endl;
        std::cout << "Mode: " << mode(preds) << std::endl;
    }
};
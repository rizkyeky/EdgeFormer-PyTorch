#include <onnxruntime_cxx_api.h>

class OnnxModel {

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Session* session;

    OnnxModel() {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        const char* model_path = "model.onnx";
        session = new Ort::Session(env, model_path, session_options);
    }

    // run inference
    void run() {
        // input
        std::vector<int64_t> input_shape = {1, 3, 224, 224};
        std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());
        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};
        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        // output
        auto output_tensor = output_tensors.front().GetTensorMutableData<float>();
        std::vector<float> output_data(output_tensor, output_tensor + 1000);

        auto output_data = output_tensors[0].GetTensorMutableData<float>();
        std::vector<int64_t> output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    }
};
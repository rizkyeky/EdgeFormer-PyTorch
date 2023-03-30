// #include <onnxruntime_cxx_api.h>

// class OnnxModel {
//     OnnxModel() {
//         Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
//         Ort::SessionOptions session_options;
//         session_options.SetIntraOpNumThreads(1);
//         const char* model_path = "model.onnx";
//         Ort::Session session(env, model_path, session_options);
//     }
// };
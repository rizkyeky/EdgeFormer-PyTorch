cmake_minimum_required(VERSION 3.0.0)
project(test_opencv VERSION 0.1.0)

include(CTest)
enable_testing()

set(ONNXRUNTIME_ROOTDIR "/opt/homebrew/Cellar/onnxruntime/1.14.1")

find_package(TorchVision REQUIRED)
find_package(OpenCV REQUIRED)

include_directories("${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")
link_libraries("${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.dylib")

set(SOURCES main.cpp detector.cpp classifier.cpp onnx.cpp)
add_executable(main ${SOURCES})

target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main TorchVision::TorchVision)

target_compile_options(main PUBLIC -std=c++20)
target_compile_features(main PUBLIC cxx_range_for)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

cmake_minimum_required(VERSION 3.0.0)
project(test_opencv VERSION 0.1.0)

# include(CTest)
# enable_testing()

if(LINUX) 
    set(ROOTDIR "/home/labai/eky_edgeformer/eky")
    set(OpenCV_DIR "${ROOTDIR}/opencv")
    set(ONNXRUNTIME_DIR "${ROOTDIR}/onnxruntime")
    set(Torch_DIR "${ROOTDIR}/libtorch/share/cmake/Torch")
    set(TORCH_DIR "${ROOTDIR}/libtorch")
    set(TorchVision_DIR "${ROOTDIR}/torchvision")

    include_directories("${TORCH_DIR}/include")

    include_directories("${TorchVision_DIR}/include")
    link_libraries("${TorchVision_DIR}/libtorchvision.so")

    include_directories("${ONNXRUNTIME_DIR}/include")
    link_libraries("${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
else()
    set(ROOTDIR "/opt/homebrew/Caskroom/miniforge/base/envs/cv")
    set(OpenCV_DIR "${ROOTDIR}/lib/cmake/opencv4")
    set(ONNXRUNTIME_DIR "/opt/homebrew/Cellar/onnxruntime/1.14.1")
    
    find_package(TorchVision REQUIRED)

    include_directories("${ONNXRUNTIME_DIR}/include/onnxruntime/core/session")
    link_libraries("${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
endif()


find_package(OpenCV REQUIRED)

include_directories("${OpenCV_INCLUDE_DIRS}")
find_library(OpenCV_LIBS opencv_core HINTS "${ROOTDIR}/lib")
    
# set(TORCH_DIR "${ROOTDIR}/lib/python3.9/site-packages/torch")
# include_directories("${TORCH_DIR}/include/torch")
# link_libraries("${TORCH_DIR}/lib/libtorch.dylib")

set(SOURCES main.cpp detector.cpp classifier.cpp onnx.cpp)
add_executable(main ${SOURCES})

if(LINUX) 
    target_link_libraries(main ${TorchVision_LIBS})
else()
    target_link_libraries(main TorchVision::TorchVision)
endif()

target_link_libraries(main ${OpenCV_LIBS})

target_compile_options(main PUBLIC -std=c++20)
target_compile_features(main PUBLIC cxx_range_for)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

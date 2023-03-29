cmake_minimum_required(VERSION 3.0.0)
project(test_opencv VERSION 0.1.0)

include(CTest)
enable_testing()

# include_directories(/Users/eky/Dev/libtorch/include)

# set(CMAKE_PREFIX_PATH "/Users/eky/Dev/libtorch")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -arch arm64")

# find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(NumCpp REQUIRED)

# find_library(LIBTORCH_LIBRARY torch PATHS "/Users/eky/Dev/libtorch/lib")

# add_executable(testmodel main.cpp)
set(SOURCES main.cpp detector.cpp classifier.cpp)
add_executable(testmodel ${SOURCES})

target_link_libraries(testmodel ${OpenCV_LIBS})
# target_link_libraries(testmodel PUBLIC ${TORCH_LIBRARIES})
# target_link_libraries(testmodel PUBLIC ${TORCHVISION_LIBRARIES})
target_link_libraries(testmodel TorchVision::TorchVision)
# target_link_libraries(testob NumCpp::NumCpp)

# target_include_directories(testmodel PUBLIC ${TORCH_INCLUDE})
# target_include_directories(testmodel PUBLIC ${TORCHVISION_INCLUDE})

target_compile_options(testmodel PUBLIC -std=c++20)
target_compile_features(testmodel PUBLIC cxx_range_for)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

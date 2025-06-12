cmake_minimum_required(VERSION 3.10)
project(open_aimforce)

set(CMAKE_CXX_STANDARD 17)

set(TF_LIB_DIR ~/built-binaries/tensorflow/bin)
set(TF_INC_DIR ../model_dev/venv/lib/python3.10/site-packages/tensorflow/include)

include_directories(
        .
        ${TF_INC_DIR}
        ${TF_INC_DIR}/external/nsync/public
        ${TF_INC_DIR}/external/com_google_absl
        ${TF_INC_DIR}/external/eigen_archive
        ${TF_INC_DIR}/external
        ${TF_INC_DIR}/tensorflow
        ${TF_INC_DIR}/tsl
        /usr/include/eigen3
        ./cnpy
)

link_directories(${TF_LIB_DIR})

# 1. Shared library target
add_library(model_predictor SHARED
        model_predictor.cc
        model_predictor.hpp
)

# 2. Main executable links to shared lib
add_executable(open_aimforce
        main.cc
)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(model_predictor
            tensorflow_cc
            tensorflow_framework
            pthread
            OpenMP::OpenMP_CXX
    )
else()
    target_link_libraries(model_predictor
            tensorflow_cc
            tensorflow_framework
            pthread
    )
endif()

target_link_libraries(open_aimforce
        model_predictor
)

set_target_properties(model_predictor PROPERTIES
        BUILD_RPATH "${TF_LIB_DIR}"
        INSTALL_RPATH "${TF_LIB_DIR}"
)
set_target_properties(open_aimforce PROPERTIES
        BUILD_RPATH "${TF_LIB_DIR}"
        INSTALL_RPATH "${TF_LIB_DIR}"
)

cmake_minimum_required(VERSION 3.26)

project(tensorrt_upgrade_guard LANGUAGES CXX CUDA)

option(UPGRADE_GUARD_BUILD_TESTS "Build focused CUDA and plugin tests" ON)
option(UPGRADE_GUARD_BUILD_FAULTS "Build quarantined seeded fault targets" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES must come from the environment lock")
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(TensorRT REQUIRED)
find_package(CUDAToolkit REQUIRED)
if(UPGRADE_GUARD_BUILD_TESTS AND NOT TARGET CUDA::nvtx3)
  message(FATAL_ERROR "The CUDA Toolkit NVTX3 target is required for profiling evidence")
endif()

add_library(
  upgrade_guard_residual_rmsnorm SHARED
  cpp/plugin/residual_rmsnorm_plugin.cpp
  cpp/plugin/residual_rmsnorm_creator.cpp
  cpp/plugin/register_plugin.cpp
  cpp/kernels/residual_rmsnorm_naive.cu
  cpp/kernels/residual_rmsnorm_optimized.cu
)
target_include_directories(upgrade_guard_residual_rmsnorm PRIVATE cpp)
target_link_libraries(
  upgrade_guard_residual_rmsnorm
  PRIVATE TensorRT::nvinfer TensorRT::nvinfer_plugin CUDA::cudart
)
target_compile_options(
  upgrade_guard_residual_rmsnorm
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wconversion;-Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--Werror=cross-execution-space-call>
)
set_target_properties(
  upgrade_guard_residual_rmsnorm
  PROPERTIES OUTPUT_NAME upgrade_guard_residual_rmsnorm
)

if(UPGRADE_GUARD_BUILD_TESTS)
  enable_testing()
  add_executable(
    upgrade_guard_kernel_tests
    cpp/tests/kernel_tests.cu
    cpp/kernels/residual_rmsnorm_naive.cu
    cpp/kernels/residual_rmsnorm_optimized.cu
  )
  target_include_directories(upgrade_guard_kernel_tests PRIVATE cpp)
  target_link_libraries(upgrade_guard_kernel_tests PRIVATE TensorRT::nvinfer CUDA::cudart)
  add_test(NAME residual_rmsnorm_kernels COMMAND upgrade_guard_kernel_tests)
  add_executable(upgrade_guard_plugin_smoke cpp/tests/plugin_smoke.cpp)
  target_include_directories(upgrade_guard_plugin_smoke PRIVATE cpp)
  target_link_libraries(
    upgrade_guard_plugin_smoke
    PRIVATE upgrade_guard_residual_rmsnorm TensorRT::nvinfer CUDA::cudart
  )
  add_test(NAME residual_rmsnorm_plugin_smoke COMMAND upgrade_guard_plugin_smoke)
  add_executable(
    upgrade_guard_kernel_benchmark
    cpp/tests/kernel_benchmark.cu
    cpp/kernels/residual_rmsnorm_naive.cu
    cpp/kernels/residual_rmsnorm_optimized.cu
  )
  target_include_directories(upgrade_guard_kernel_benchmark PRIVATE cpp)
  target_link_libraries(
    upgrade_guard_kernel_benchmark
    PRIVATE TensorRT::nvinfer CUDA::cudart CUDA::nvtx3
  )
endif()

if(UPGRADE_GUARD_BUILD_FAULTS)
  add_library(
    upgrade_guard_fault_plugin SHARED
    cpp/plugin/residual_rmsnorm_plugin.cpp
    cpp/plugin/residual_rmsnorm_creator.cpp
    cpp/plugin/register_plugin.cpp
    cpp/kernels/residual_rmsnorm_naive.cu
    cpp/kernels/residual_rmsnorm_optimized.cu
    cpp/faults/omitted_residual_fault.cu
  )
  target_compile_definitions(
    upgrade_guard_fault_plugin PRIVATE UPGRADE_GUARD_FAULT_OMIT_RESIDUAL_HIDDEN=259
  )
  target_include_directories(upgrade_guard_fault_plugin PRIVATE cpp)
  target_link_libraries(
    upgrade_guard_fault_plugin
    PRIVATE TensorRT::nvinfer TensorRT::nvinfer_plugin CUDA::cudart
  )
  target_compile_options(
    upgrade_guard_fault_plugin
    PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wconversion;-Werror>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--Werror=cross-execution-space-call>
  )
  add_executable(upgrade_guard_tail_oob_fault cpp/faults/tail_oob_fault.cu)
  target_link_libraries(upgrade_guard_tail_oob_fault PRIVATE CUDA::cudart)
  add_executable(
    upgrade_guard_gpu_faults
    cpp/faults/gpu_faults.cu
  )
  target_include_directories(upgrade_guard_gpu_faults PRIVATE cpp)
  target_link_libraries(
    upgrade_guard_gpu_faults
    PRIVATE upgrade_guard_fault_plugin TensorRT::nvinfer CUDA::cudart
  )
  add_executable(
    upgrade_guard_serialization_fault
    cpp/faults/serialization_fault.cpp
  )
  target_include_directories(upgrade_guard_serialization_fault PRIVATE cpp)
  target_link_libraries(
    upgrade_guard_serialization_fault
    PRIVATE upgrade_guard_residual_rmsnorm TensorRT::nvinfer CUDA::cudart
  )
endif()

find_path(
  TensorRT_INCLUDE_DIR
  NvInfer.h
  HINTS ${TensorRT_ROOT} $ENV{TensorRT_ROOT}
  PATH_SUFFIXES include include/x86_64-linux-gnu
  PATHS /usr /usr/local /opt/tensorrt
)

find_library(
  TensorRT_NVINFER_LIBRARY
  nvinfer
  HINTS ${TensorRT_ROOT} $ENV{TensorRT_ROOT}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
  PATHS /usr /usr/local /opt/tensorrt
)

find_library(
  TensorRT_PLUGIN_LIBRARY
  nvinfer_plugin
  HINTS ${TensorRT_ROOT} $ENV{TensorRT_ROOT}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
  PATHS /usr /usr/local /opt/tensorrt
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  TensorRT
  REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_NVINFER_LIBRARY TensorRT_PLUGIN_LIBRARY
)

if(TensorRT_FOUND AND NOT TARGET TensorRT::nvinfer)
  add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
  set_target_properties(
    TensorRT::nvinfer
    PROPERTIES
      IMPORTED_LOCATION "${TensorRT_NVINFER_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
  )
  add_library(TensorRT::nvinfer_plugin UNKNOWN IMPORTED)
  set_target_properties(
    TensorRT::nvinfer_plugin
    PROPERTIES
      IMPORTED_LOCATION "${TensorRT_PLUGIN_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
  )
endif()

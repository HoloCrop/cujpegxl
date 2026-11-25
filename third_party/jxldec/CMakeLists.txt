cmake_minimum_required(VERSION 3.24)
project(camera_coefficients LANGUAGES C)

include(FetchContent)
FetchContent_Declare(jxldec
    URL https://codeload.github.com/kjk/jxldec/tar.gz/b062dd5b34a76f29a858b2772a003b4bcd6c8b2a
    URL_HASH SHA256=dcb90d32317087e1f11e5a374161a17bc1e5888476a267490384eddabaaccf0a
)
FetchContent_MakeAvailable(jxldec)

# Generate a private copy for the coefficient reader. This also works with a
# read-only source tree supplied via FETCHCONTENT_SOURCE_DIR_JXLDEC.
find_program(PATCH_EXECUTABLE patch REQUIRED)
set(reader "${CMAKE_CURRENT_BINARY_DIR}/jxldec_camera.inc")
set(upstream "${jxldec_SOURCE_DIR}/dist/jxl.c")
set(hooks "${CMAKE_CURRENT_SOURCE_DIR}/camera-coefficients.patch")
add_custom_command(OUTPUT "${reader}"
    COMMAND "${PATCH_EXECUTABLE}" --batch --fuzz=0 --output "${reader}" "${upstream}" "${hooks}"
    DEPENDS "${upstream}" "${hooks}"
    VERBATIM
)

add_library(camera_coefficients STATIC ../../src/camera_coefficients.c "${reader}")
target_include_directories(camera_coefficients PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(camera_coefficients PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_VISIBILITY_PRESET hidden
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

cmake_minimum_required(VERSION 3.20)
project(KittenI VERSION 0.5.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(KITTENI_BUILD_TESTS "Build KittenI unit tests" ON)
option(KITTENI_BUILD_CLI "Build KittenI CLI" ON)
option(KITTENI_BUILD_SHARED "Build KittenI C shared library (libkitteni)" ON)
option(KITTENI_AVX2 "Enable AVX2 SIMD optimizations" ON)

if(MSVC)
    add_compile_options(/W4 /O2 /EHsc /utf-8)
    if(KITTENI_AVX2)
        add_compile_options(/arch:AVX2)
    endif()
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
    if(KITTENI_AVX2 AND (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686"))
        add_compile_options(-mavx2 -mfma -mf16c -mbmi2)
    endif()
endif()

set(KITTENI_SOURCES
    src/core/json.cpp
    src/core/mmap_loader.cpp
    src/core/hub_cache.cpp
    src/kernels/cpu_features.cpp
    src/kernels/ternary_unpack.cpp
    src/kernels/fwht_emb4.cpp
    src/kernels/ternary_gemv.cpp
    src/kernels/nn_ops.cpp
    src/kernels/dsp_fft.cpp
    src/kernels/bitnet_ops.cpp
    src/frontend/bpe_tokenizer.cpp
    src/frontend/text_normalizer.cpp
    src/frontend/audio_joiner.cpp
    src/lm/model_loader.cpp
    src/lm/sampler.cpp
    src/lm/qwen3_ternary.cpp
    src/vocoder/s3_flow.cpp
    src/vocoder/hift_gan.cpp
    src/vocoder/vocoder_pipeline.cpp
    src/cloning/sincnet_xvector.cpp
    src/cloning/campplus.cpp
    src/cloning/s3_tokenizer.cpp
    src/cloning/voice_cloner.cpp
    src/api/engine.cpp
    src/api/c_api.cpp
)

add_library(kitteni_core STATIC ${KITTENI_SOURCES})
target_include_directories(kitteni_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

find_package(Threads REQUIRED)
target_link_libraries(kitteni_core PUBLIC Threads::Threads)

if(KITTENI_BUILD_SHARED)
    add_library(kitteni_shared SHARED src/api/c_api.cpp)
    target_compile_definitions(kitteni_shared PRIVATE KITTENI_SHARED_BUILD)
    target_link_libraries(kitteni_shared PRIVATE kitteni_core)
    set_target_properties(kitteni_shared PROPERTIES OUTPUT_NAME "kitteni")
    if(MINGW)
        target_link_options(kitteni_shared PRIVATE -static-libgcc -static-libstdc++)
    endif()
endif()

if(KITTENI_BUILD_CLI)
    add_executable(kitteni cli/main.cpp)
    target_link_libraries(kitteni PRIVATE kitteni_core)
endif()

include(GNUInstallDirs)
install(DIRECTORY include/kitteni DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS kitteni_core ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(KITTENI_BUILD_SHARED)
    install(TARGETS kitteni_shared
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()
if(KITTENI_BUILD_CLI)
    install(TARGETS kitteni RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(KITTENI_BUILD_TESTS)
    enable_testing()
    add_executable(test_phase1 tests/test_phase1.cpp)
    target_link_libraries(test_phase1 PRIVATE kitteni_core)
    add_test(NAME Phase1Tests COMMAND test_phase1)

    add_executable(test_phase2 tests/test_phase2.cpp)
    target_link_libraries(test_phase2 PRIVATE kitteni_core)
    add_test(NAME Phase2Tests COMMAND test_phase2)

    add_executable(test_phase3 tests/test_phase3.cpp)
    target_link_libraries(test_phase3 PRIVATE kitteni_core)
    add_test(NAME Phase3Tests COMMAND test_phase3)

    add_executable(test_phase4 tests/test_phase4.cpp)
    target_link_libraries(test_phase4 PRIVATE kitteni_core)
    add_test(NAME Phase4Tests COMMAND test_phase4)

    add_executable(test_phase5 tests/test_phase5.cpp)
    target_link_libraries(test_phase5 PRIVATE kitteni_core)
    add_test(NAME Phase5Tests COMMAND test_phase5)
endif()

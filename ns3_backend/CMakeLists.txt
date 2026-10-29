cmake_minimum_required(VERSION 3.20)
project(dooly_ns3_backend CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Point CMAKE_PREFIX_PATH (or ns3_DIR) at the ns-3 install prefix produced by
# setup_ns3.sh, e.g. -DCMAKE_PREFIX_PATH=$HOME/ns-3-dev/install
find_package(ns3 REQUIRED)

add_executable(dooly-collective dooly-collective.cc)
target_include_directories(dooly-collective PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(dooly-collective PRIVATE ns3::core ns3::network ns3::point-to-point)

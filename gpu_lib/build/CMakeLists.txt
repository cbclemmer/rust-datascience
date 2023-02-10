cmake_minimum_required(VERSION 3.18.4)
project(gpu_lib C)
add_library(lib STATIC ../lib.c)
install(TARGETS lib DESTINATION .)
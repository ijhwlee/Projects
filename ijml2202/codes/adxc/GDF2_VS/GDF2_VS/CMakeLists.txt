# CMakeList.txt: GDF2_VS에 대한 CMake 프로젝트, 여기에 소스를 포함하고
# 프로젝트 특정 논리를 정의합니다.
#
cmake_minimum_required (VERSION 3.8)

# 이 프로젝트의 실행 파일에 소스를 추가합니다.
add_executable (GDF2_VS "GDF2_VS.cpp" "GDF2_VS.h" "cprogram.cc" "cprogram.h" "doublestack.cc" "doublestack.h" "errorfunction.cc" "errorfunction.h" "fparser.cpp" "fparser.hh" "gdfprogram.cpp" "gdfprogram.h" "get_options.cc" "get_options.h" "multi_population.cc" "multi_population.h" "population.cc" "population.h" "program.cpp" "program.h" "rule.cc" "rule.h" "symbol.cc" "symbol.h" "udf.cc" "udf.h" "getopt.cpp" "getopt.h" )

# TODO: 필요한 경우 테스트를 추가하고 대상을 설치합니다.

#!/bin/bash

CTP_ROOT=$1

cpplint --recursive \
--exclude="${CTP_ROOT}/include/hermes_shm/constants/singleton_macros.h" \
--exclude="${CTP_ROOT}/include/hermes_shm/data_structures/internal/template" \
--exclude="${CTP_ROOT}/include/hermes_shm/data_structures/internal/shm_container_macro.h" \
--exclude="${CTP_ROOT}/src/singleton.cc" \
--exclude="${CTP_ROOT}/src/data_structure_singleton.cc" \
--exclude="${CTP_ROOT}/include/hermes_shm/util/formatter.h" \
--exclude="${CTP_ROOT}/include/hermes_shm/util/errors.h" \
"${CTP_ROOT}/src" "${CTP_ROOT}/include" "${CTP_ROOT}/test"
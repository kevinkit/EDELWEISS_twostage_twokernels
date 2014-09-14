cmake_minimum_required(VERSION 2.8)
find_package(PkgConfig)
pkg_check_modules(GLIB2 glib-2.0>=2.26 REQUIRED)


set(KERNELS "edel_kernel_one.cl" "edel_kernel_two.cl")
set(BINARIES "" )
set(DEPS  "-lOpenCL")


include_directories(${GLIB2_INCLUDE_DIRS})
list(APPEND DEPS ${GLIB2_LIBRARIES})
list(APPEND BINARIES "edel_two_kernel" )


foreach (BINARY ${BINARIES})
    add_executable(${BINARY} ${BINARY}.c ocl.c)

    target_link_libraries(${BINARY} ${DEPS})
endforeach ()


#foreach (KERNEL ${KERNELS})
#    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL}
#                   ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL})
#endforeach ()


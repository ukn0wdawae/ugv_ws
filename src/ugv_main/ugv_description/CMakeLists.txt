cmake_minimum_required(VERSION 3.5)

project(ugv_description)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# legacy_code/ only sorts the source tree: its folders are installed where they always were
install(DIRECTORY meshes legacy_code/launch legacy_code/urdf legacy_code/rviz
    DESTINATION share/${PROJECT_NAME}
)

install(FILES model.config
    DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(ament_cmake)
ament_export_dependencies(ament_cmake_python)
ament_export_dependencies(rosidl_default_generators)

ament_package()


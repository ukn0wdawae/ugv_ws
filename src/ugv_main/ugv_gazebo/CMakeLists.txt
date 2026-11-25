cmake_minimum_required(VERSION 3.5)

project(ugv_gazebo)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# legacy_code/ only sorts the source tree: its files are installed next to the live ones, so
# `ros2 launch ugv_gazebo <file>` and every share/ugv_gazebo/... path stay the same.
install(DIRECTORY models legacy_code/models legacy_code/rviz legacy_code/maps
    DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY launch/legacy_code/nav launch/legacy_code/nav_bringup launch/legacy_code/slam_nav
    DESTINATION share/${PROJECT_NAME}/launch
)
foreach(dir launch launch/bringup launch/slam urdf worlds param)
  file(GLOB files CONFIGURE_DEPENDS ${dir}/*.* ${dir}/legacy_code/*.*)
  install(FILES ${files} DESTINATION share/${PROJECT_NAME}/${dir})
endforeach()

install(PROGRAMS scripts/spawn_entity_clean.py
    DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(ament_cmake)
ament_export_dependencies(ament_cmake_python)
ament_export_dependencies(rosidl_default_generators)

ament_package()


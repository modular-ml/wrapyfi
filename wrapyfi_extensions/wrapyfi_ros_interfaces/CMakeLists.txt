cmake_minimum_required(VERSION 3.0.2)
project(wrapyfi_ros_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(catkin REQUIRED COMPONENTS
   roscpp
   rospy
   std_msgs
   message_generation
)

add_message_files(
  FILES
  ROSAudioMessage.msg
)

add_service_files(
  FILES
  ROSAudioService.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
 LIBRARIES wrapyfi_ros_interfaces
 CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
 DEPENDS system_lib
)



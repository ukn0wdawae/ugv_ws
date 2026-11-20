FROM osrf/ros:humble-desktop

ARG DEBIAN_FRONTEND=noninteractive

# Gazebo Classic 11 comes via gazebo-ros-pkgs (not available on Ubuntu 24 host).
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-gazebo-ros-pkgs \
      ros-humble-gazebo-ros \
      ros-humble-gazebo-plugins \
      ros-humble-robot-state-publisher \
      ros-humble-joint-state-publisher \
      ros-humble-joint-state-publisher-gui \
      ros-humble-xacro \
      ros-humble-teleop-twist-keyboard \
      ros-humble-rviz2 \
      ros-humble-navigation2 \
      ros-humble-nav2-bringup \
      ros-humble-slam-toolbox \
      ros-humble-rtabmap-ros \
      ros-humble-robot-localization \
      python3-colcon-common-extensions \
      python3-pip \
      python3-numpy \
      libgl1-mesa-dri \
      mesa-utils \
      x11-apps \
      unzip \
      nano \
    && rm -rf /var/lib/apt/lists/*

ENV QT_X11_NO_MITSHM=1
# Avoid ~30s stall looking up models online
ENV GAZEBO_MODEL_DATABASE_URI=""
ENV UGV_MODEL=ugv_rover

RUN mkdir -p /tmp/runtime-root && chmod 0700 /tmp/runtime-root \
 && mkdir -p /root/.gazebo/models

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc && \
    echo '[ -f /root/ugv_ws/install/setup.bash ] && source /root/ugv_ws/install/setup.bash' >> /root/.bashrc && \
    echo 'export UGV_MODEL=${UGV_MODEL:-ugv_rover}' >> /root/.bashrc && \
    echo 'export GAZEBO_MODEL_PATH=/root/ugv_ws/gazebo_models:/root/.gazebo/models:/root/ugv_ws/src/ugv_main/ugv_gazebo/models:${GAZEBO_MODEL_PATH}:/usr/share/gazebo-11/models' >> /root/.bashrc

WORKDIR /root/ugv_ws
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

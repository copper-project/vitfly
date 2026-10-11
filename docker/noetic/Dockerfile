FROM osrf/ros:noetic-desktop-full

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG USER_ID=1000
ARG GROUP_ID=1000

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        git \
        libeigen3-dev \
        libgl1-mesa-glx \
        libglu1-mesa \
        libgtk2.0-0 \
        libopencv-dev \
        libyaml-cpp-dev \
        libzmqpp-dev \
        mesa-utils \
        python3-catkin-tools \
        python3-matplotlib \
        python3-pandas \
        python3-pip \
        python3-scipy \
        python3-yaml \
        unzip \
        xauth \
    && rm -rf /var/lib/apt/lists/*

# The rollout scripts import the model even in privileged-expert mode. Keep
# this container CPU-only: native Arch PyTorch owns training on the host GPU.
RUN python3 -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        typing-extensions==4.12.2 \
        torch==2.4.1 \
        torchvision==0.19.1 \
    && python3 -m pip install --no-cache-dir \
        pandas==2.0.3 \
        scipy==1.10.1 \
        uniplot==0.12.8

# dodgelib declares this catkin package, but Ubuntu Noetic does not publish it.
RUN git clone https://github.com/ethz-asl/benchmark_catkin.git /opt/benchmark_catkin \
    && git -C /opt/benchmark_catkin checkout 5018b4bb92ee0021a8f5a2884660a1e02c4573d4

RUN if getent group "${GROUP_ID}" >/dev/null; then \
        group_name="$(getent group "${GROUP_ID}" | cut -d: -f1)"; \
    else \
        group_name=vitfly; \
        groupadd --gid "${GROUP_ID}" "${group_name}"; \
    fi \
    && useradd --uid "${USER_ID}" --gid "${group_name}" --create-home --shell /bin/bash vitfly \
    && mkdir -p /ws/src /ws/build /ws/devel /ws/logs \
    && ln -s /opt/benchmark_catkin /ws/src/benchmark_catkin \
    && chown -R "${USER_ID}:${GROUP_ID}" /ws

COPY catkin-build.bash /usr/local/bin/vitfly-catkin-build
RUN chmod 0755 /usr/local/bin/vitfly-catkin-build

ENV FLIGHTMARE_PATH=/ws/src/vitfly/flightmare
ENV MPLBACKEND=Agg
ENV NVIDIA_DRIVER_CAPABILITIES=all

USER vitfly
WORKDIR /ws/src/vitfly

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]

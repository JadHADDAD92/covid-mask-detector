FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

# Configure Timezone so apt doesn't hang
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Create the user
ARG USERNAME=mluser
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    # Add sudo support.
    apt-get update && \
    apt-get install -y sudo ffmpeg python3-pip git libsm6 libxext6 && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Create symlinks
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install pip packages
COPY pyproject.toml /app/pyproject.toml
WORKDIR /app
RUN pip3 install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

USER $USERNAME
ENV SHELL /bin/bash

ARG VARIANT
FROM mcr.microsoft.com/devcontainers/base:${VARIANT}

# --- Configure environment ---

# Configure DEBIAN_FRONTEND to avoid warnings
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3-pip pep8 pylint

RUN pip3 install matplotlib tabulate scipy

# --- Set up user ---
USER vscode
WORKDIR /home/vscode/workspace

# --- Set up entrypoint ---

SHELL ["/bin/bash", "-c"]
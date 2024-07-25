
ARG BASE=python:3.12-slim

ARG OPENFOAM_VERSION=2406
FROM opencfd/openfoam-default:${OPENFOAM_VERSION} AS openfoam-com

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get -y install --no-install-recommends \
    python3-pip \
 && (python3 -m pip install --upgrade --no-cache-dir pip || true) \
 && rm -rf /var/lib/apt/lists/*


ARG OPENFOAM_VERSION=12
FROM ubuntu:24.04 AS openfoam-org

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get -y install --no-install-recommends \
    wget \
    software-properties-common \
 && sh -c "wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc" \
 && add-apt-repository http://dl.openfoam.org/ubuntu \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get -y install --no-install-recommends \
    python3-pip \
    openfoam${OPENFOAM_VERSION} \
 && rm -rf /var/lib/apt/lists/*


FROM ${BASE}

COPY . /src/

RUN python3 -m pip install --no-cache-dir --break-system-packages /src || python3 -m pip install --no-cache-dir /src \
 && rm -rf /src \
# smoke test
 && python3 -c 'import foamlib'

CMD ["python3"]

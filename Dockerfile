
ARG BASE=python:3.12-slim

ARG OPENFOAM_VERSION=2406
FROM microfluidica/openfoam:${OPENFOAM_VERSION} AS openfoam

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get -y install --no-install-recommends \
    python3-pip \
 && (python3 -m pip install --upgrade --no-cache-dir pip || true) \
 && rm -rf /var/lib/apt/lists/*


FROM ${BASE}

COPY . /src/

RUN (python3 -m pip install --no-cache-dir --break-system-packages /src || python3 -m pip install --no-cache-dir /src) \
 && rm -rf /src \
# smoke test
 && python3 -c 'import foamlib'

CMD ["python3"]

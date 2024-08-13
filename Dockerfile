
ARG BASE=python:3.12-slim

ARG OPENFOAM_VERSION=2406
FROM microfluidica/openfoam:${OPENFOAM_VERSION} AS openfoam

ARG VIRTUAL_ENV=/opt/venv

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get -y install --no-install-recommends \
    python3-venv \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv ${VIRTUAL_ENV}

ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

CMD ["python3"]


FROM ${BASE}

COPY . /src/

RUN pip install --no-cache-dir /src

RUN rm -rf /src \
# smoke test
 && python -c 'import foamlib'

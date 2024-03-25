FROM opencfd/openfoam-default:2312

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get -y install --no-install-recommends \
    python3-pip \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir --upgrade pip

COPY . /src/

RUN pip install --no-cache-dir /src \
 && rm -rf /src \
# smoke test
 && python3 -c 'import foamlib'

CMD ["python3"]

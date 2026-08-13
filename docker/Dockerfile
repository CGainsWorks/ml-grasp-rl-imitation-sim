# CPU image. Everything in this repository trains on a CPU; the GPU buys
# nothing for 128-unit networks and MuJoCo's physics is single-threaded per
# environment anyway.
#
# Rendering is deliberately not set up here: offscreen GL in a container needs
# either EGL with the host driver passed through or a software rasteriser, and
# both are environment-specific enough that pretending otherwise in a Dockerfile
# would be a lie. Run `make videos` on the host.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MUJOCO_GL=disable \
    OMP_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        make \
        libgl1 \
        libglfw3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -m pytest tests/ -q

CMD ["python", "scripts/offline_demo.py"]

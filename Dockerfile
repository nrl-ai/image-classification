FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
RUN mkdir -p /workspace

# Configure library
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PYTHONPATH=/workspace

COPY requirements.txt /requirements.txt
RUN python -m pip install --upgrade pip && python -m pip install -r /requirements.txt

COPY . /workspace
WORKDIR /workspace

CMD ["bash"]

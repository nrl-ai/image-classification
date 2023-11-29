FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
WORKDIR /workspace

# Configure library
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PYTHONPATH=/workspace

COPY requirements.txt /requirements.txt
RUN python -m pip install --upgrade pip && python -m pip install -r /requirements.txt

COPY ./pretrains /workspace/
COPY *.py /workspace/
COPY *.sh /workspace/

CMD ["bash"]

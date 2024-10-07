# # Use the official NVIDIA TensorFlow image with GPU support and CUDA 12.4
# FROM nvcr.io/nvidia/tensorflow:23.08-tf2-py3

# # Set the working directory inside the container
# WORKDIR /workspace

# # Copy your code into the container
# COPY game_state.py /workspace/
# COPY mcts.py /workspace/
# COPY train_utils.py /workspace/
# COPY training_loop.py /workspace/

# # Install any additional Python dependencies if necessary
# # Uncomment and adjust if you have other dependencies
# # RUN pip install -r requirements.txt

# # Set the entrypoint to run the training loop
# ENTRYPOINT ["python", "training_loop.py"]

# old tensorflow for old cuda
# Use an older NVIDIA TensorFlow image (e.g., TensorFlow 2.7 with CUDA 11)
FROM nvcr.io/nvidia/tensorflow:21.09-tf2-py3

# Set the working directory inside the container
WORKDIR /workspace

# Copy your code into the container
COPY game_state.py /workspace/
COPY mcts.py /workspace/
COPY train_utils.py /workspace/
COPY training_loop.py /workspace/

# Set the entrypoint to run the training loop
ENTRYPOINT ["python", "training_loop.py"]

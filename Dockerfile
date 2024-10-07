# Use the official TensorFlow 2.9 with CUDA 11.2 image
FROM tensorflow/tensorflow:2.9.0-gpu

# Set the working directory inside the container
WORKDIR /workspace

# Copy your Python code into the container
COPY game_state.py /workspace/
COPY mcts.py /workspace/
COPY train_utils.py /workspace/
COPY training_loop.py /workspace/

# Set the entrypoint to run the training loop
ENTRYPOINT ["python", "training_loop.py"]

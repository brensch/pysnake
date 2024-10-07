```bash
docker run --gpus all -it --rm \
 -v $(pwd):/workspace \
 -w /workspace \
 tensorflow/tensorflow:2.9.0-gpu \
 python training_loop.py
```

### Detached

```bash
docker run --gpus all -d --rm \
 -v $(pwd):/workspace \
 -w /workspace \
 tensorflow/tensorflow:2.9.0-gpu \
 python -u training_loop.py
```

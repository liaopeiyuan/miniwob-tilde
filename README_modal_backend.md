# MiniWoB Modal Backend

This directory contains a Modal-based serverless backend for MiniWoB that allows you to run browser automation tasks using serverless Chrome instances in the cloud.

## Overview

The Modal backend provides an alternative to the local Selenium-based backend (`selenium_instance.py`). Instead of running Chrome locally, it spins up serverless Chrome instances using Modal's cloud infrastructure. This offers several advantages:

- **Scalability**: Can handle many parallel instances without local resource constraints.
- **Reproducibility**: Consistent environment across different machines.
- **Cost Efficiency**: Only pay for compute time when instances are active.
- **No Local Setup**: No need to install Chrome or ChromeDriver locally.

## Architecture

The Modal backend is integrated directly into the `MiniWoBEnvironment`. You can select it by passing `backend="modal"` to `gymnasium.make()`.

```
Your Code                  ┌──────────────────────────────────┐
                           │       MiniWoBEnvironment         │
                           │                                  │
gym.make(..., backend)─────►  if backend == "modal":           │
                           │      use ModalInstance           │
                           │  else:                           │
                           │      use SeleniumInstance        │
                           └──────────────────────────────────┘
```

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install "miniwob-plusplus[modal]"
    ```
    This will install `miniwob`, `gymnasium`, and `modal`.

2.  **Configure Modal Authentication**:
    ```bash
    modal token new
    ```
    Follow the instructions to authenticate with your Modal account.

3.  **Deploy Modal App**: The first time you run the Modal backend, it will automatically build the required Docker image and deploy the app to Modal's servers. This might take a few minutes.

## Usage

Using the Modal backend is as simple as specifying `backend="modal"` when creating the environment. The library handles the Modal application lifecycle automatically.

### Basic Usage

```python
import gymnasium as gym
import miniwob  # Registers the environments

# --- Using the 'modal' backend ---
print("🚀 --- Testing 'modal' (serverless Chrome) backend --- 🚀")
env = gym.make(
    "miniwob/click-test-v1",
    backend="modal",  # Select the modal backend
)
# Use the environment as usual.
# The Modal app starts automatically with the environment...
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
# ...and shuts down automatically when you close it.
env.close()


# --- Using the default 'selenium' backend ---
print("🚀 --- Testing 'selenium' (local Chrome) backend --- 🚀")
env_local = gym.make(
    "miniwob/click-test-v1",
    backend="selenium",   # Explicitly select selenium
    render_mode="human"   # Show the browser window locally
)
# ...
```

### Running the Example

The `example_modal_usage.py` script demonstrates how to switch between backends.

```bash
python example_modal_usage.py
```

## Key Differences from Selenium Backend

### Advantages
- **No Local Chrome Installation**: Chrome runs in Modal's cloud environment.
- **Better Scalability**: Can run many instances in parallel.
- **Consistent Environment**: Same Chrome version and environment every time.
- **Automatic Cleanup**: Modal handles resource cleanup.

### Considerations
- **Network Latency**: Remote calls add some latency (~0.5-1s per step/reset) compared to local execution.
- **Internet Dependency**: Requires an internet connection to Modal's servers.
- **Modal Account Required**: Need a Modal account and authentication.

### API Compatibility
The `backend="modal"` parameter makes the `MiniWoBEnvironment` use the `ModalInstance` internally. The environment's public API remains identical. You can switch between backends without changing the rest of your code.

## Configuration Options

All `gymnasium.make` arguments are passed to the `MiniWoBEnvironment` constructor.

- `backend`: "modal" or "selenium".
- `render_mode`: "human" is only supported for the "selenium" backend.
- `action_space_config`, `base_url`, `wait_ms`, etc., are all supported.

## Performance Tips

1.  **Reuse Environments**: The Modal container stays warm between calls, so reusing an `env` object is more efficient than calling `gym.make()` repeatedly.
2.  **Container Idle Timeout**: The default idle timeout is 5 minutes. Containers are automatically cleaned up after this period.
3.  **Parallel Environments**: You can create multiple environments with `backend="modal"` to run tasks in parallel.

## Debugging

To see logs from your Modal app:

```bash
modal logs miniwob-serverless
```

You can also check the Modal dashboard at [modal.com](https://modal.com) for container status and logs.

## Cost Considerations

Modal charges based on compute time. A typical MiniWoB container uses:
- **CPU**: ~1 vCPU
- **Memory**: ~2GB RAM
- **Storage**: ~1GB for the Docker image

Costs are generally very low for development and testing. Refer to Modal's pricing for details.

## Limitations

1.  **No GUI**: `render_mode="human"` is not supported for the Modal backend.
2.  **Increased Latency**: Each `env.step()` and `env.reset()` involves a network round-trip.
3.  **Browser Extensions**: Disabled for security and performance.

## Troubleshooting

### Common Issues

1.  **"Driver not initialized"**: This should be handled internally by the environment. If it occurs, it's likely a bug.
2.  **Network timeouts**: Check your internet connection and the Modal service status.
3.  **Authentication errors**: Ensure you have run `modal token new`.

### Getting Help

-   Modal Documentation: [modal.com/docs](https://modal.com/docs)
-   Modal Community Slack: [modal.com/slack](https://modal.com/slack)
-   MiniWoB++ Issues: [github.com/Farama-Foundation/miniwob-plusplus/issues](https://github.com/Farama-Foundation/miniwob-plusplus/issues) 
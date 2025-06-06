#!/usr/bin/env python3
"""
Test the parallel execution of 64 MiniWoB environments using Modal and multiprocessing.
"""
import gymnasium as gym
import logging
import multiprocessing
import time
from typing import cast

# Import miniwob to register the environments
import miniwob
from miniwob.action import ActionTypes
from miniwob.environment import MiniWoBEnvironment


def run_env(env_id: int, seed: int):
    """
    Function executed by each process.
    It creates, runs, and closes a single MiniWoB environment.
    """
    # Configure logging for each process to include the environment ID
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [Env-{env_id:02d}] %(message)s",
        force=True,
    )

    env = None
    try:
        logging.info("Creating environment: miniwob/click-test-v1")
        # Each process will create its own environment, triggering a new Modal session
        env = gym.make("miniwob/click-test-v1", backend="modal")

        logging.info(f"Resetting environment with seed {seed}")
        obs, info = env.reset(seed=seed)
        logging.info(f"Got utterance: {obs['utterance']}")

        # Create a deterministic click action
        action = cast(MiniWoBEnvironment, env.unwrapped).create_action(
            ActionTypes.CLICK_COORDS, coords=[80, 105]  # Near the center button
        )

        logging.info("Executing action...")
        step_start_time = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        step_end_time = time.time()

        logging.info(
            f"Action executed in {step_end_time - step_start_time:.2f}s. "
            f"Reward: {reward:.2f}, Terminated: {terminated}"
        )
        return True

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return False

    finally:
        if env:
            logging.info("Closing environment.")
            env.close()


def main():
    """
    Main function to set up and run the parallel test.
    """
    NUM_ENVS = 64
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [Main] %(message)s"
    )

    logging.info(f"Starting test with {NUM_ENVS} parallel environments.")

    # Set the multiprocessing start method to 'spawn'.
    # This is crucial for compatibility with Modal, as 'fork' can lead to
    # issues with shared client connections and background threads.
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # It can only be set once.
        pass

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=NUM_ENVS) as pool:
        # Generate arguments (env_id, seed) for each process
        args = [(i, 42 + i) for i in range(NUM_ENVS)]

        total_start_time = time.time()
        logging.info("Dispatching tasks to the process pool...")
        results = pool.starmap(run_env, args)
        total_end_time = time.time()

    num_successful = sum(1 for r in results if r)
    logging.info(f"--- Test Summary ---")
    logging.info(f"Total environments: {NUM_ENVS}")
    logging.info(f"Successful runs: {num_successful}")
    logging.info(f"Failed runs: {NUM_ENVS - num_successful}")
    logging.info(
        f"All {NUM_ENVS} environments finished in {total_end_time - total_start_time:.2f}s."
    )

if __name__ == "__main__":
    main() 
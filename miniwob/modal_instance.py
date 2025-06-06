from __future__ import annotations

"""Modal-based interface for MiniWoB using serverless Chrome instances."""
import json
import logging
import pathlib
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import pickle
import base64
import contextlib

import numpy as np
import modal

if TYPE_CHECKING:
    from selenium.webdriver.remote.webdriver import WebDriver

from miniwob.action import Action, ActionSpaceConfig
from miniwob.constants import (
    FLIGHT_TASK_HEIGHT,
    FLIGHT_TASK_WIDTH,
    FLIGHT_WINDOW_HEIGHT,
    FLIGHT_WINDOW_WIDTH,
    TASK_HEIGHT,
    TASK_WIDTH,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from miniwob.dom import DOMElement
from miniwob.fields import FieldExtractor, get_field_extractor
from miniwob.http_server import start_http_server
from miniwob.observation import (
    Observation,
    create_empty_observation,
    create_empty_screenshot,
    create_observation,
)
from miniwob.reward import RewardProcessor, get_original_reward
from miniwob.screenshot import pil_to_numpy_array


HTML_DIR = pathlib.Path(__file__).parent / "html"
DEFAULT_BASE_URL = f"file://{HTML_DIR}/miniwob/"

# Modal App Configuration
app = modal.App("miniwob-serverless")

# Chrome + Selenium image
chrome_image = modal.Image.debian_slim(python_version="3.12").run_commands(
    "apt-get update",
    "apt-get install -y git wget gnupg2 software-properties-common curl unzip jq",
    # Install Google Chrome
    "wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add -",
    'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list',
    "apt-get update",
    "apt-get install -y google-chrome-stable",
    # Clone the repo and install from source to include all package data (like the html files).
    # This makes the image self-contained and removes dependency on the local filesystem.
    "git clone https://github.com/liaopeiyuan/miniwob-tilde.git /root/miniwob-tilde",
    "pip install /root/miniwob-tilde",
    # Install other Python dependencies
    "pip install numpy selenium>=4.0.0",
)


@app.cls(image=chrome_image, scaledown_window=300)
class ModalBrowserSession:
    """Modal class that manages a Chrome browser session."""

    driver: Optional['WebDriver'] = None
    inner_width: Optional[int] = None
    inner_height: Optional[int] = None
    current_url: Optional[str] = None
    task_width: Optional[int] = None
    task_height: Optional[int] = None
    # Store creation parameters for recreation
    subdomain: Optional[str] = None
    headless: Optional[bool] = None
    SYNC_SCREEN_ID: str = "sync-task-cover"
    RESET_BLOCK_SLEEP_TIME: float = 0.05
    RESET_BLOCK_MAX_ATTEMPT: int = 20
    httpd: Optional[Any] = None
    base_url: Optional[str] = None

    @modal.enter()
    def setup(self):
        """Initialize Chrome driver when container starts."""
        from selenium import webdriver
        from selenium.common.exceptions import TimeoutException
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.wait import WebDriverWait

        logging.info("Setting up Chrome driver in Modal container")
        self._start_http_server()

    def _start_http_server(self):
        """Starts an HTTP server inside the container."""
        import functools
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
        from threading import Thread

        # Because we install the package from source, we can find the html dir via the library path.
        import miniwob
        html_dir = pathlib.Path(miniwob.__file__).parent / "html"

        self.httpd = ThreadingHTTPServer(
            ("localhost", 0),
            functools.partial(SimpleHTTPRequestHandler, directory=str(html_dir)),
        )

        def serve_forever(server):
            with server:
                server.serve_forever()

        thread = Thread(target=serve_forever, args=(self.httpd,))
        thread.daemon = True
        thread.start()
        address, port = self.httpd.server_address
        self.base_url = f"http://{address}:{port}/"
        logging.info(f"HTTP server started at {self.base_url}")

    @modal.exit()
    def teardown(self):
        """Clean up Chrome driver when container exits."""
        if self.driver:
            try:
                self.driver.quit()
                logging.info("Chrome driver closed successfully")
            except Exception as e:
                logging.error(f"Error closing Chrome driver: {e}")
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            logging.info("HTTP server shut down successfully")

    def _create_driver_internal(self):
        """Internal logic to create the Chrome driver, with retries."""
        from selenium import webdriver
        from selenium.common.exceptions import TimeoutException
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.wait import WebDriverWait

        assert self.base_url is not None
        assert self.subdomain is not None
        if self.subdomain.startswith("flight."):
            path = self.subdomain.replace(".", "/") + "/wrapper.html"
            self.current_url = urllib.parse.urljoin(self.base_url, path)
        else:
            path = f"miniwob/{self.subdomain}.html"
            self.current_url = urllib.parse.urljoin(self.base_url, path)

        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-logging")
        options.add_argument("--silent")

        if not self.headless:
            options.add_argument("--app=" + self.current_url)

        # Retry loop for robustness against transient startup errors.
        for attempt in range(3):
            try:
                driver = webdriver.Chrome(options=options)
                driver.implicitly_wait(5)
                driver.get(self.current_url)
                logging.info(f"WebDriver getting URL: {self.current_url}")
                WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.ID, self.SYNC_SCREEN_ID))
                )
                self.inner_width, self.inner_height = driver.execute_script(
                    "return [window.innerWidth, window.innerHeight];"
                )
                self.driver = driver
                logging.info(f"Successfully created Chrome driver on attempt {attempt + 1}")
                return
            except Exception as e:
                logging.warning(
                    f"Failed to create Chrome driver on attempt {attempt + 1}/3: {e}"
                )
                if 'driver' in locals() and driver:
                    driver.quit()
                if attempt == 2:
                    logging.error("All attempts to create driver failed.")
                time.sleep(1)

    @modal.method()
    def create_driver(
        self, subdomain: str, headless: bool, task_width: int, task_height: int
    ):
        """Create and initialize the Chrome driver."""
        if self.driver is not None:
            return True
        # Store parameters for potential recreation
        self.subdomain = subdomain
        self.headless = headless
        self.task_width = task_width
        self.task_height = task_height
        self._create_driver_internal()
        return self.driver is not None

    def _ensure_driver(self):
        """Checks if the driver exists and attempts to recreate it if not."""
        if self.driver is None:
            logging.warning("Driver not found. Attempting to recreate it.")
            self._create_driver_internal()
            if self.driver is None:
                raise RuntimeError("Driver is not initialized and could not be recreated.")

    @modal.method()
    def execute_script(self, script: str):
        """Execute JavaScript on the page."""
        self._ensure_driver()
        assert self.driver is not None
        return self.driver.execute_script(script)

    @modal.method()
    def execute_action(self, action_data: str):
        """Execute a selenium action (action_data is base64 encoded pickle)."""
        from miniwob.selenium_actions import execute_action_on_chromedriver
        self._ensure_driver()
        # Deserialize the action data
        action, fields, config = pickle.loads(base64.b64decode(action_data))
        execute_action_on_chromedriver(action, fields, config, self.driver)
        return True

    @modal.method()
    def get_screenshot_data(self):
        """Get screenshot as base64 encoded image data."""
        from miniwob.screenshot import get_screenshot, pil_to_numpy_array
        self._ensure_driver()
        if self.inner_width is None or self.inner_height is None:
            raise RuntimeError("Driver dimensions not set")

        if self.task_width is None or self.task_height is None:
            raise RuntimeError("Task dimensions not set")

        img = get_screenshot(
            self.driver,
            true_width=self.inner_width,
            true_height=self.inner_height,
            crop_width=self.task_width,
            crop_height=self.task_height,
        )
        img_array = pil_to_numpy_array(img)
        # Encode numpy array as base64
        return base64.b64encode(pickle.dumps(img_array)).decode('utf-8')

    @modal.method()
    def refresh_page(self):
        """Refresh the current page."""
        self._ensure_driver()
        assert self.driver is not None
        self.driver.get(self.current_url)
        return True

    @modal.method()
    def close_driver(self):
        """Close the Chrome driver."""
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
                return True
            except Exception as e:
                logging.error(f"Error closing driver: {e}")
                return False
        return True


class ModalInstance:
    """Interface between Python and Chrome via Modal serverless functions."""

    def __init__(
        self,
        index: int,
        subdomain: str,
        headless: bool = True,  # Default to headless for serverless
        base_url: Optional[str] = None,
        field_extractor: Optional[FieldExtractor] = None,
        reward_processor: Optional[RewardProcessor] = None,
        wait_ms: float = 0.0,
        block_on_reset: bool = True,
        refresh_freq: int = 0,
        data_mode: str = "train",
    ):
        """Initialize a Modal-based MiniWoB instance.
        
        Args:
            index: Instance index
            subdomain: MiniWoB task name (e.g., "click-test")
            headless: Whether to render GUI (always True for serverless)
            base_url: Base URL for MiniWoB tasks
            field_extractor: Function that extracts fields from utterance
            reward_processor: Function that processes rewards
            wait_ms: Pause time after each action in milliseconds
            block_on_reset: Block until page loads on reset
            refresh_freq: Frequency of page refreshes
            data_mode: Data mode (e.g., "train", "test")
        """
        self.index = index
        self.headless = True  # Always headless for serverless
        self.died = False
        self.subdomain = subdomain
        
        # Configure URL and dimensions based on subdomain
        if subdomain.startswith("flight."):
            if not base_url:
                base_url = start_http_server(str(HTML_DIR))
            assert not base_url.startswith("file://"), (
                "For {} domain, MINIWOB_BASE_URL cannot be file://."
            ).format(subdomain)
            self.url = urllib.parse.urljoin(
                base_url, subdomain.replace(".", "/") + "/wrapper.html"
            )
            self.window_width = FLIGHT_WINDOW_WIDTH
            self.window_height = FLIGHT_WINDOW_HEIGHT
            self.task_width = FLIGHT_TASK_WIDTH
            self.task_height = FLIGHT_TASK_HEIGHT
        else:
            if not base_url:
                # For file:// URLs, we need to serve via HTTP for Modal
                base_url = start_http_server(str(HTML_DIR))
            if base_url.startswith("file://"):
                # Convert file:// to http:// for Modal
                base_url = start_http_server(str(HTML_DIR))
            self.url = urllib.parse.urljoin(base_url, f"{subdomain}.html")
            self.window_width = WINDOW_WIDTH
            self.window_height = WINDOW_HEIGHT
            self.task_width = TASK_WIDTH
            self.task_height = TASK_HEIGHT
            
        # Initialize field extractor and reward processor
        if not field_extractor:
            self.field_extractor = get_field_extractor(subdomain)
        else:
            self.field_extractor = field_extractor
        if not reward_processor:
            self.reward_processor = get_original_reward
        else:
            self.reward_processor = reward_processor
            
        self.wait_ms = wait_ms
        self.block_on_reset = block_on_reset
        self.refresh_freq = refresh_freq
        self.num_episodes = 0
        self.mode = data_mode
        self.record_screenshots = True
        self.start_time = float("inf")
        self.cached_fields = []
        
        # Initialize Modal browser session
        self.browser_session = None
        self._exit_stack = contextlib.ExitStack()
        self._app_is_running = False
        
    def start(self):
        """Start the instance (creates driver and starts Modal app)."""
        if self.browser_session is not None:
            return  # Already started
        if not self._app_is_running:
            logging.info("Starting Modal app context...")
            self._exit_stack.enter_context(app.run())
            self._app_is_running = True
            logging.info("Modal app is running.")
        self.browser_session = ModalBrowserSession()
        success = self.browser_session.create_driver.remote(
            self.subdomain, self.headless, self.task_width, self.task_height
        )
        if not success:
            raise RuntimeError(f"Failed to create driver for instance {self.index}")

    def create_driver(self):
        """DEPRECATED: start() now handles driver creation."""
        pass
    
    def close(self):
        """Close the Modal browser session and stop the app."""
        if self.browser_session:
            try:
                # This call might fail if the app is already stopping
                self.browser_session.close_driver.remote()
            except Exception as e:
                logging.warning(f"Error during remote driver close: {e}")
        self.died = True
        self._exit_stack.close()
        self._app_is_running = False
        logging.info(f"Modal app for instance {self.index} shut down.")
    
    def reset(self, obs: List[Any], infos: List[Any], seed: Any):
        """Reset the instance for a new episode."""
        if self.refresh_freq:
            assert seed is not None, "reset() must specify seed if refresh_freq is specified"
            
        i = self.index
        
        # Create driver if not exists
        if not self._app_is_running:
            # This should not be reached if start() is called correctly.
            logging.warning("Modal app not running. Calling start() from reset().")
            self.start()

        self.force_stop()
        self.begin_task(seed=seed)
        obs[i], extra_metadata = self.get_observation(use_cached_fields=False)
        metadata = self.get_metadata()
        metadata.update(extra_metadata)
        infos[i] = metadata
    
    def step(
        self,
        action: Optional[Action],
        action_space_config: ActionSpaceConfig,
        obs: List[Any],
        rewards: List[Any],
        dones: List[Any],
        infos: List[Any],
    ):
        """Execute a step with the given action."""
        i = self.index
        self.perform(action, action_space_config)
        metadata = self.get_metadata()
        rewards[i] = self.reward_processor(metadata)
        dones[i] = metadata["done"]
        if metadata["done"]:
            obs[i] = self.get_empty_observation()
            extra_metadata = {}
        else:
            obs[i], extra_metadata = self.get_observation(use_cached_fields=True)
        metadata["elapsed"] = max(0.0, time.time() - self.start_time)
        metadata.update(extra_metadata)
        infos[i] = metadata
    
    def force_stop(self):
        """Force stop the current task."""
        if not self.browser_session:
            return
        self.browser_session.execute_script.remote("return core.endEpisode(0);")
    
    def begin_task(self, seed: Any = None):
        """Begin a new task episode."""
        if not self.browser_session:
            raise RuntimeError("Browser session not initialized")
            
        self.num_episodes += 1
        if self.refresh_freq and self.num_episodes % self.refresh_freq == 0:
            self.browser_session.refresh_page.remote()
        
        if seed is not None:
            self.set_seed(seed)
        self.set_mode(self.mode)
        self.browser_session.execute_script.remote("core.startEpisodeReal();")
        
        if self.block_on_reset:
            for _ in range(20):  # RESET_BLOCK_MAX_ATTEMPT
                if self.browser_session.execute_script.remote("return WOB_TASK_READY;"):
                    break
                time.sleep(0.05)  # RESET_BLOCK_SLEEP_TIME
            else:
                raise RuntimeError(f"Instance {self.index} does not load properly")
        elif self.wait_ms:
            time.sleep(self.wait_ms / 1000.0)
        
        self.start_time = time.time()
    
    def perform(self, action: Optional[Action], action_space_config: ActionSpaceConfig):
        """Perform an action."""
        if not self.browser_session:
            raise RuntimeError("Browser session not initialized")
            
        if action is not None:
            if self.get_metadata()["done"]:
                logging.warning(
                    "Cannot call %s on instance %d, which is already done",
                    action,
                    self.index,
                )
            else:
                # Serialize action data for remote execution
                action_data = base64.b64encode(
                    pickle.dumps((action, self.cached_fields, action_space_config))
                ).decode('utf-8')
                self.browser_session.execute_action.remote(action_data)
        
        if self.wait_ms:
            time.sleep(self.wait_ms / 1000.0)
    
    def get_empty_observation(self) -> Observation:
        """Get an empty observation for terminated sessions."""
        return create_empty_observation(self.task_width, self.task_height)
    
    def get_observation(
        self, use_cached_fields: bool = False
    ) -> Tuple[Observation, Dict[str, Any]]:
        """Get the current observation."""
        if not self.browser_session:
            raise RuntimeError("Browser session not initialized")
            
        # Get the utterance
        response = self.browser_session.execute_script.remote("return core.getUtterance();")
        if isinstance(response, dict):
            utterance = response["utterance"]
        else:
            utterance = response
            
        if use_cached_fields:
            fields = self.cached_fields
        else:
            if isinstance(response, dict):
                fields = list(response["fields"].items())
            else:
                fields = self.field_extractor(utterance)
            self.cached_fields = fields
        
        # Get the DOM
        dom_info = self.browser_session.execute_script.remote("return core.getDOMInfo();")
        root_dom = DOMElement(dom_info)
        
        # Get screenshot if requested
        if self.record_screenshots:
            img_data = self.browser_session.get_screenshot_data.remote()
            img = pickle.loads(base64.b64decode(img_data))
            logging.info(f"Instance {self.index}: Successfully rendered image of size {img.shape}")
        else:
            img = create_empty_screenshot(self.task_width, self.task_height)
        
        observation = create_observation(utterance, root_dom, img, fields)
        return observation, {"root_dom": root_dom}
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get current metadata."""
        if not self.browser_session:
            raise RuntimeError("Browser session not initialized")
            
        return self.browser_session.execute_script.remote(
            "return {"
            '"done": WOB_DONE_GLOBAL,'
            '"env_reward": WOB_REWARD_GLOBAL,'
            '"raw_reward": WOB_RAW_REWARD_GLOBAL,'
            '"reason": WOB_REWARD_REASON,'
            "};"
        )
    
    def visualize_attention(self, attention: Optional[np.ndarray]):
        """Visualize attention weights."""
        if attention is None or not self.browser_session:
            return
            
        # Encode as JSON
        if isinstance(attention, np.ndarray):
            attention = attention.tolist()
        encoded = json.dumps(attention)
        
        # Send to the driver
        self.browser_session.execute_script.remote(f"core.visualizeAttention({encoded});")
    
    def set_seed(self, seed: Any):
        """Set the random seed."""
        if not self.browser_session:
            return
        self.browser_session.execute_script.remote(f"Math.seedrandom({repr(seed)});")
    
    def set_mode(self, mode: str):
        """Set the data mode."""
        if not self.browser_session:
            return
        self.browser_session.execute_script.remote(f'core.setDataMode("{mode}");')
    
    def call(self, func, *args):
        """Call function directly (no threading in Modal)."""
        func(*args)
    
    def wait(self):
        """No-op for thread compatibility."""
        pass 
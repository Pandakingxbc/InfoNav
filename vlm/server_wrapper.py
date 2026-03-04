import base64
import os
import random
import socket
import time
from typing import Any, Dict

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request

try:
    from waitress import serve
    WAITRESS_AVAILABLE = True
except ImportError:
    WAITRESS_AVAILABLE = False


class ServerMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def process_payload(self, payload: dict) -> dict:
        raise NotImplementedError


def host_model(model: Any, name: str, port: int = 5000) -> None:
    """
    Hosts a model as a REST API using Flask.
    """
    app = Flask(__name__)

    # Disable Flask response caching to prevent memory buildup
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.config['JSON_AS_ASCII'] = False

    @app.route(f"/{name}", methods=["POST"])
    def process_request() -> Dict[str, Any]:
        payload = request.json
        result = model.process_payload(payload)

        # Explicitly delete payload to free memory
        del payload

        return jsonify(result)

    # Use production WSGI server for better performance and Keep-Alive support
    if WAITRESS_AVAILABLE:
        print(f"Using waitress WSGI server (supports HTTP Keep-Alive)")
        serve(
            app,
            host="127.0.0.1",
            port=port,
            threads=8,  # Number of worker threads
            channel_timeout=3600,  # Keep connections alive for 1 hour (prevent premature closure)
            cleanup_interval=60,  # Clean up idle connections every 60 seconds
            recv_bytes=65536,  # Receive buffer size (64KB for large payloads)
            send_bytes=65536,  # Send buffer size
        )
    else:
        print(f"Warning: waitress not available, using Flask dev server (no Keep-Alive)")
        # Fallback to Flask development server
        app.run(
            host="localhost",
            port=port,
            threaded=True,  # Enable multi-threading
            use_reloader=False,  # Disable auto-reloader (causes issues in production)
            debug=False  # Disable debug mode for better performance
        )


def bool_arr_to_str(arr: np.ndarray) -> str:
    """Converts a boolean array to a string."""
    packed_str = base64.b64encode(arr.tobytes()).decode()
    return packed_str


def str_to_bool_arr(s: str, shape: tuple) -> np.ndarray:
    """Converts a string to a boolean array."""
    # Convert the string back into bytes using base64 decoding
    bytes_ = base64.b64decode(s)

    # Convert bytes to np.uint8 array
    bytes_array = np.frombuffer(bytes_, dtype=np.uint8)

    # Reshape the data back into a boolean array
    unpacked = bytes_array.reshape(shape)
    return unpacked


def image_to_str(img_np: np.ndarray, quality: float = 90.0) -> str:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    retval, buffer = cv2.imencode(".jpg", img_np, encode_param)
    img_str = base64.b64encode(buffer).decode("utf-8")
    return img_str


def str_to_image(img_str: str) -> np.ndarray:
    img_bytes = base64.b64decode(img_str)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_np = cv2.imdecode(img_arr, cv2.IMREAD_ANYCOLOR)
    return img_np


# Global HTTP session for connection pooling and reuse
_http_sessions = {}

def get_http_session(url: str) -> requests.Session:
    """
    Get or create a persistent HTTP session for the given URL.
    This enables connection pooling and reduces overhead.
    """
    if url not in _http_sessions:
        session = requests.Session()
        # Configure session for better performance
        session.headers.update({"Connection": "keep-alive"})
        # Set connection pool size
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=0
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        _http_sessions[url] = session

    return _http_sessions[url]

def send_request(url: str, **kwargs: Any) -> dict:
    response = {}
    for attempt in range(10):
        try:
            response = _send_request(url, **kwargs)
            break
        except Exception as e:
            if attempt == 9:
                print(e)
                exit()
            else:
                print(f"Error: {e}. Retrying in 20-30 seconds...")
                time.sleep(20 + random.random() * 10)

    return response


def _send_request(url: str, **kwargs: Any) -> dict:
    lockfiles_dir = "lockfiles"
    if not os.path.exists(lockfiles_dir):
        os.makedirs(lockfiles_dir)
    filename = url.replace("/", "_").replace(":", "_") + ".lock"
    filename = filename.replace("localhost", socket.gethostname())
    filename = os.path.join(lockfiles_dir, filename)
    try:
        while True:
            # Use a while loop to wait until this filename does not exist
            while os.path.exists(filename):
                # If the file exists, wait 50ms and try again
                time.sleep(0.001)

                try:
                    # If the file was last modified more than 120 seconds ago, delete it
                    if time.time() - os.path.getmtime(filename) > 120:
                        os.remove(filename)
                except FileNotFoundError:
                    pass

            rand_str = str(random.randint(0, 1000000))

            with open(filename, "w") as f:
                f.write(rand_str)
            time.sleep(0.001)
            try:
                with open(filename, "r") as f:
                    if f.read() == rand_str:
                        break
            except FileNotFoundError:
                pass

        # Create a payload dict which is a clone of kwargs but all np.array values are
        # converted to strings
        payload = {}
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                payload[k] = image_to_str(v, quality=kwargs.get("quality", 90))
            else:
                payload[k] = v
        # Set the headers
        headers = {"Content-Type": "application/json"}

        # Use persistent session for connection reuse
        session = get_http_session(url)

        start_time = time.time()
        retry_count = 0
        while True:
            try:
                # Use session instead of requests directly for connection pooling
                resp = session.post(url, headers=headers, json=payload, timeout=20)
                if resp.status_code == 200:
                    result = resp.json()
                    # Don't call resp.close() - let the session manage connection pooling
                    # The connection will be automatically returned to the pool for reuse
                    break
                else:
                    raise Exception("Request failed")
            except (
                requests.exceptions.Timeout,
                requests.exceptions.RequestException,
                requests.exceptions.ConnectionError,
            ) as e:
                print(f"Connection error: {e}")

                # If connection was reset/closed, clear the session cache and retry
                if retry_count < 3:
                    print(f"Recreating session and retrying ({retry_count + 1}/3)...")
                    if url in _http_sessions:
                        _http_sessions[url].close()
                        del _http_sessions[url]
                    session = get_http_session(url)
                    retry_count += 1
                    time.sleep(0.5)  # Brief delay before retry
                    continue

                if time.time() - start_time > 20:
                    raise Exception("Request timed out after 20 seconds")

        try:
            # Delete the lock file
            os.remove(filename)
        except FileNotFoundError:
            pass

        # Explicitly delete payload to free memory immediately
        del payload

    except Exception as e:
        try:
            # Delete the lock file
            os.remove(filename)
        except FileNotFoundError:
            pass
        # Clean up payload even on error
        if 'payload' in locals():
            del payload
        raise e

    return result

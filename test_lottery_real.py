import sys
import os
import logging
import ssl
import requests
import urllib3

# Disable warnings for insecure requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Add current directory to path
sys.path.append(os.getcwd())

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SSL Hack (Monkeypatch requests) ---
# Force verify=False for all requests
old_request = requests.Session.request
def new_request(self, method, url, *args, **kwargs):
    kwargs['verify'] = False
    return old_request(self, method, url, *args, **kwargs)
requests.Session.request = new_request

# Also patch top-level API if used directly (though usually libraries use Session or internal calls)
# But patching Session.request is usually enough for requests>=2.x as top level calls use sessions internally
# ----------------

try:
    from my_commands.lottery_gpt import lottery_gpt
    print("Successfully imported lottery_gpt")
except ImportError as e:
    print(f"Failed to import lottery_gpt: {e}")
    # Try to fix path if needed, but sys.path is already set
    sys.exit(1)

def test_real_crawler():
    print("\n--- Testing lottery_gpt('大樂透') ---")
    try:
        result = lottery_gpt("大樂透")
        print("Result preview:")
        print(result[:500])
    except Exception as e:
        print(f"lottery_gpt failed: {e}")

if __name__ == "__main__":
    test_real_crawler()

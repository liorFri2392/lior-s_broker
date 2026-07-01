"""github_secret.py - Single implementation of the PORTFOLIO_JSON secret sync.

Previously ``_try_update_github_secret`` / ``_update_secret_via_api`` were
duplicated verbatim in both ``portfolio_analyzer`` and ``deposit_advisor``. Both
now call :func:`update_portfolio_secret`.

Two strategies are attempted, in order:
  1. the GitHub CLI (``gh secret set``) if it is installed;
  2. the GitHub REST API using ``GITHUB_TOKEN`` + libsodium sealed-box
     encryption (requires ``requests`` and ``PyNaCl``).

All failures are swallowed and reported via the return value (best-effort sync).
"""
import base64
import json
import logging
import subprocess
from typing import Dict

logger = logging.getLogger(__name__)

DEFAULT_REPO = "liorFri2392/lior-s_broker"
SECRET_NAME = "PORTFOLIO_JSON"

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from nacl import encoding, public
    HAS_PYNACL = True
except ImportError:
    HAS_PYNACL = False


def _update_via_api(token: str, portfolio: Dict, repo: str) -> bool:
    """Update the secret through the GitHub REST API (sealed-box encrypted)."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    try:
        pk_url = f"https://api.github.com/repos/{repo}/actions/secrets/public-key"
        resp = requests.get(pk_url, headers=headers, timeout=10)
        resp.raise_for_status()
        pk = resp.json()

        portfolio_json = json.dumps(portfolio, ensure_ascii=False, indent=2)
        pk_obj = public.PublicKey(pk["key"].encode("utf-8"), encoding.Base64Encoder())
        encrypted = public.SealedBox(pk_obj).encrypt(portfolio_json.encode("utf-8"))
        encrypted_value = base64.b64encode(encrypted).decode("utf-8")

        secret_url = f"https://api.github.com/repos/{repo}/actions/secrets/{SECRET_NAME}"
        payload = {"encrypted_value": encrypted_value, "key_id": pk["key_id"]}
        resp = requests.put(secret_url, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("✅ GitHub secret updated automatically (via GitHub API)")
        return True
    except Exception as e:  # noqa: BLE001 - best-effort
        logger.debug(f"GitHub API secret update failed: {e}")
        return False


def update_portfolio_secret(portfolio: Dict, repo: str = DEFAULT_REPO,
                            verbose: bool = False) -> bool:
    """Best-effort sync of the portfolio to the ``PORTFOLIO_JSON`` GitHub secret.

    Returns True on success, False if neither the CLI nor the API path worked.
    Never raises.
    """
    import os

    # Method 1: GitHub CLI.
    try:
        result = subprocess.run(
            ["gh", "--version"], capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            portfolio_json = json.dumps(portfolio, ensure_ascii=False, indent=2)
            proc = subprocess.Popen(
                ["gh", "secret", "set", SECRET_NAME, "--repo", repo],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True,
            )
            proc.communicate(input=portfolio_json, timeout=10)
            if proc.returncode == 0:
                logger.info("✅ GitHub secret updated automatically (via GitHub CLI)")
                if verbose:
                    print("   ✅ GitHub secret updated automatically!")
                return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # CLI not available; fall through to the API path.

    # Method 2: GitHub REST API.
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token and HAS_REQUESTS and HAS_PYNACL:
        if _update_via_api(github_token, portfolio, repo):
            if verbose:
                print("   ✅ GitHub secret updated automatically!")
            return True

    return False

"""
CreditLens-AI
Frontend API Client

All HTTP communication with the FastAPI backend lives here.
Transport failures and HTTP errors are normalised into ``APIError``
with a message that is safe to show to users; the raw server detail
is kept separately for a "technical details" view.

The client never invents endpoints. ``get_capabilities`` reads the
backend's own OpenAPI document so the UI can tell which optional
endpoints the connected backend actually provides.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlsplit

import requests

from config import API_BASE_URL, API_TIMEOUT


logger = logging.getLogger(__name__)

# Endpoints (method, path relative to the API prefix).
EP_PREDICT = ("POST", "/predict")
EP_ASSESS = ("POST", "/assess/{customer_id}")
EP_SAMPLE_CUSTOMERS = ("GET", "/customers/sample")
EP_HISTORY = ("GET", "/customers/{customer_id}/predictions")
EP_DASHBOARD = ("GET", "/dashboard/summary")

# FastAPI's response body for a route that does not exist.
_UNKNOWN_ROUTE_DETAIL = "Not Found"


class APIError(Exception):
    """User-presentable error from the CreditLens API client."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        detail: str | None = None,
        endpoint_missing: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail
        self.endpoint_missing = endpoint_missing


class CreditLensAPI:
    """
    Client for the CreditLens-AI FastAPI backend.
    """

    def __init__(
        self,
        base_url: str = API_BASE_URL,
        timeout: int = API_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ========================================================
    # Transport
    # ========================================================

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        action: str = "complete the request",
    ) -> Any:

        url = f"{self.base_url}{path}"

        try:
            response = requests.request(
                method,
                url,
                json=json,
                params=params,
                timeout=self.timeout,
            )

        except requests.exceptions.ConnectionError as exc:
            logger.warning("API unreachable: %s", exc)
            raise APIError(
                "CreditLens-AI API is currently unavailable. "
                "Please verify that the backend service is running "
                "and has finished starting.",
                detail=f"Could not connect to {self.base_url}",
            ) from exc

        except requests.exceptions.Timeout as exc:
            logger.warning("API timeout: %s", url)
            raise APIError(
                "The assessment is taking longer than expected. "
                "Please try again.",
                detail=f"No response from {url} within "
                       f"{self.timeout} seconds.",
            ) from exc

        except requests.exceptions.RequestException as exc:
            logger.warning("API request failed: %s", exc)
            raise APIError(f"Unable to {action}.") from exc

        if not response.ok:
            raise self._http_error(response, action)

        try:
            return response.json()
        except ValueError as exc:
            raise APIError(
                "The risk service returned an unexpected response.",
                status_code=response.status_code,
                detail=f"Non-JSON response from {url}",
            ) from exc

    @staticmethod
    def _http_error(
        response: requests.Response,
        action: str,
    ) -> APIError:

        status = response.status_code

        detail: str | None = None

        try:
            body = response.json().get("detail")
            detail = body if isinstance(body, str) else None
        except (ValueError, AttributeError):
            detail = None

        if status == 404 and detail == _UNKNOWN_ROUTE_DETAIL:
            return APIError(
                "This feature is not available on the connected API. "
                "The backend is probably an older version that does "
                "not provide this endpoint.",
                status_code=status,
                detail=f"HTTP 404 for {response.request.method} "
                       f"{response.url}",
                endpoint_missing=True,
            )

        if status == 422:
            return APIError(
                "Please review the information entered before "
                "submitting the assessment.",
                status_code=status,
                detail=detail,
            )

        if status == 400:
            return APIError(
                "The risk models could not accept this customer's "
                "data. Please review the data and try again.",
                status_code=status,
                detail=detail,
            )

        if detail:
            return APIError(
                detail if status in (404, 503) else
                f"{detail} See the backend console for details.",
                status_code=status,
                detail=f"HTTP {status}: {detail}",
            )

        return APIError(
            f"Unable to {action} (status {status}).",
            status_code=status,
        )

    # ========================================================
    # Capability detection
    # ========================================================

    def get_capabilities(self) -> set[tuple[str, str]] | None:
        """
        Return the (METHOD, path) pairs the backend advertises in its
        OpenAPI document, with the API prefix removed, or ``None`` when
        the document cannot be read (callers should then assume the
        endpoint exists and rely on normal error handling).
        """

        parts = urlsplit(self.base_url)
        root = f"{parts.scheme}://{parts.netloc}"
        prefix = parts.path.rstrip("/")

        try:
            response = requests.get(
                f"{root}/openapi.json",
                timeout=min(self.timeout, 10),
            )
            response.raise_for_status()
            paths = response.json()["paths"]
        except (requests.RequestException, ValueError, KeyError):
            return None

        capabilities: set[tuple[str, str]] = set()

        for path, methods in paths.items():
            relative = (
                path[len(prefix):] if prefix and path.startswith(prefix)
                else path
            )
            for method in methods:
                capabilities.add((method.upper(), relative))

        return capabilities

    @staticmethod
    def supports(
        capabilities: set[tuple[str, str]] | None,
        endpoint: tuple[str, str],
    ) -> bool:
        """Optimistic: unknown capabilities mean 'try it'."""

        return capabilities is None or endpoint in capabilities

    # ========================================================
    # Health
    # ========================================================

    def health(self) -> dict[str, Any]:
        return self._request(
            "GET",
            "/health",
            action="check service health",
        )

    # ========================================================
    # Prediction
    # ========================================================

    def predict(
        self,
        customer_id: str,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """Assess using caller-supplied production features."""

        return self._request(
            "POST",
            "/predict",
            json={
                "customer_id": str(customer_id),
                "features": features,
            },
            action="complete the risk assessment",
        )

    def assess_customer(
        self,
        customer_id: str,
    ) -> dict[str, Any]:
        """Assess a customer by ID; features are looked up server-side."""

        return self._request(
            "POST",
            f"/assess/{str(customer_id).strip()}",
            action="complete the risk assessment",
        )

    def get_sample_customers(
        self,
        limit: int = 20,
    ) -> list[str]:

        data = self._request(
            "GET",
            "/customers/sample",
            params={"limit": limit},
            action="load sample customers",
        )

        return list(data.get("customer_ids", []))

    # ========================================================
    # History / Dashboard
    # ========================================================

    def get_customer_history(
        self,
        customer_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:

        data = self._request(
            "GET",
            f"/customers/{str(customer_id).strip()}/predictions",
            params={"limit": limit},
            action="load customer history",
        )

        return list(data.get("predictions", []))

    def get_dashboard_summary(self) -> dict[str, Any]:

        return self._request(
            "GET",
            "/dashboard/summary",
            action="load assessment analytics",
        )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTTP Adapter
============

Concrete implementation of HttpPort for HTTP operations.

This adapter handles:
- HTTP GET and POST requests
- Retry logic with exponential backoff
- Timeout handling
- Error reporting
"""

from typing import Dict, Any, Optional
import time


try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class RequestsHttpAdapter:
    """HTTP adapter using requests library."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """Initialize HTTP adapter.
        
        Args:
            max_retries: Maximum number of retries on failure
            retry_delay: Initial delay between retries (exponential backoff)
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library not available. Install with: pip install requests")
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def get(self, url: str, headers: Optional[Dict[str, str]] = None,
            timeout: int = 30) -> Dict[str, Any]:
        """Make HTTP GET request.
        
        Args:
            url: URL to request
            headers: Optional HTTP headers
            timeout: Request timeout in seconds
            
        Returns:
            Response with 'status', 'body', 'headers' keys
            
        Raises:
            TimeoutError: If request times out
            ConnectionError: If connection fails
        """
        return self._request('GET', url, headers=headers, timeout=timeout)
    
    def post(self, url: str, data: Dict[str, Any],
             headers: Optional[Dict[str, str]] = None,
             timeout: int = 30) -> Dict[str, Any]:
        """Make HTTP POST request.
        
        Args:
            url: URL to request
            data: Data to send in request body
            headers: Optional HTTP headers
            timeout: Request timeout in seconds
            
        Returns:
            Response with 'status', 'body', 'headers' keys
            
        Raises:
            TimeoutError: If request times out
            ConnectionError: If connection fails
        """
        return self._request('POST', url, json=data, headers=headers, timeout=timeout)
    
    def _request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with retries.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL to request
            **kwargs: Additional arguments for requests library
            
        Returns:
            Response dictionary
            
        Raises:
            TimeoutError: If request times out after retries
            ConnectionError: If connection fails after retries
        """
        last_error = None
        delay = self.retry_delay
        
        for attempt in range(self.max_retries):
            try:
                response = requests.request(method, url, **kwargs)
                return {
                    'status': response.status_code,
                    'body': response.text,
                    'headers': dict(response.headers),
                    'ok': response.ok,
                }
            except requests.Timeout as e:
                last_error = TimeoutError(f"Request timed out: {url}")
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
            except requests.ConnectionError as e:
                last_error = ConnectionError(f"Connection failed: {url}")
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
            except Exception as e:
                last_error = RuntimeError(f"Unexpected error: {str(e)}")
                break  # Don't retry on unexpected errors
        
        if last_error:
            raise last_error
        
        raise RuntimeError("Request failed for unknown reason")


class MockHttpAdapter:
    """Mock HTTP adapter for testing.
    
    This adapter returns predefined responses, making it perfect for:
    - Unit tests (no network I/O)
    - Integration tests (deterministic responses)
    - Development (no external dependencies)
    """
    
    def __init__(self):
        """Initialize mock HTTP adapter."""
        self._responses: Dict[str, Dict[str, Any]] = {}
        self._default_response = {
            'status': 200,
            'body': '{}',
            'headers': {},
            'ok': True,
        }
    
    def set_response(self, url: str, response: Dict[str, Any]) -> None:
        """Set predefined response for URL.
        
        Args:
            url: URL to mock
            response: Response dictionary with 'status', 'body', 'headers' keys
        """
        self._responses[url] = response
    
    def set_default_response(self, response: Dict[str, Any]) -> None:
        """Set default response for unmocked URLs.
        
        Args:
            response: Response dictionary
        """
        self._default_response = response
    
    def get(self, url: str, headers: Optional[Dict[str, str]] = None,
            timeout: int = 30) -> Dict[str, Any]:
        """Mock HTTP GET request.
        
        Args:
            url: URL to request
            headers: Ignored in mock
            timeout: Ignored in mock
            
        Returns:
            Predefined response for URL
        """
        return self._responses.get(url, self._default_response)
    
    def post(self, url: str, data: Dict[str, Any],
             headers: Optional[Dict[str, str]] = None,
             timeout: int = 30) -> Dict[str, Any]:
        """Mock HTTP POST request.
        
        Args:
            url: URL to request
            data: Ignored in mock
            headers: Ignored in mock
            timeout: Ignored in mock
            
        Returns:
            Predefined response for URL
        """
        return self._responses.get(url, self._default_response)

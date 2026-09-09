# The MIT License (MIT)
# Copyright © 2026 qBitTensor Labs
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
import requests

from qbittensor.utils.request.jwt_manager import JWT
from qbittensor.utils.request.request_manager import RequestManager


def _jwt(*, hours=1, token="tok"):
    return JWT(
        access_token=token,
        expires_in=3600,
        expiration_date=datetime.now(timezone.utc) + timedelta(hours=hours),
    )


def _manager() -> RequestManager:
    rm = RequestManager(
        Mock(),
        base_url="https://api.example.com/v1",
        tensorauth_url="https://tensorauth.example.com",
        netuid=63,
    )
    rm._session = Mock()
    rm._jwt_manager = Mock()
    return rm


class TestRequestManagerJwt:
    def test_refresh_jwt_stores_token(self):
        rm = _manager()
        jwt = _jwt()
        rm._jwt_manager.get_jwt.return_value = jwt
        assert rm.refresh_jwt() is jwt
        assert rm.jwt is jwt

    def test_get_header_fetches_when_missing(self):
        rm = _manager()
        jwt = _jwt(token="fresh")
        rm._jwt_manager.get_jwt.return_value = jwt
        headers = rm._get_header()
        assert headers["Authorization"] == "Bearer fresh"
        rm._jwt_manager.get_jwt.assert_called_once()

    def test_get_header_reuses_unexpired_token(self):
        rm = _manager()
        rm._jwt = _jwt(token="cached")
        headers = rm._get_header()
        assert headers["Authorization"] == "Bearer cached"
        rm._jwt_manager.get_jwt.assert_not_called()

    def test_get_header_refreshes_within_expiration_buffer(self):
        rm = _manager()
        rm._jwt = _jwt(hours=0)
        rm._jwt.expiration_date = datetime.now(timezone.utc) + timedelta(seconds=30)
        fresh = _jwt(token="renewed")
        rm._jwt_manager.get_jwt.return_value = fresh
        headers = rm._get_header()
        assert headers["Authorization"] == "Bearer renewed"
        rm._jwt_manager.get_jwt.assert_called_once()


class TestRequestManagerHttp:
    def test_build_url_joins_base_and_strips_slash(self):
        rm = _manager()
        assert rm._build_url("/datapoints") == "https://api.example.com/v1/datapoints"
        rm._base_url = None
        assert rm._build_url("/token") == "token"

    def test_get_sends_jwt_and_extra_headers(self):
        rm = _manager()
        rm._jwt = _jwt()
        response = Mock(status_code=200, text="ok")
        rm._session.get.return_value = response
        out = rm.get("jobs", params={"a": 1}, additional_headers=[("X-Foo", "bar")])
        assert out is response
        kwargs = rm._session.get.call_args.kwargs
        assert kwargs["headers"]["Authorization"] == "Bearer tok"
        assert kwargs["headers"]["X-Foo"] == "bar"
        assert kwargs["params"] == {"a": 1}

    def test_get_reraises_connection_error(self):
        rm = _manager()
        rm._jwt = _jwt()
        rm._session.get.side_effect = requests.exceptions.ConnectionError("down")
        with pytest.raises(requests.exceptions.ConnectionError):
            rm.get("jobs")

    def test_post_and_patch_hit_session(self):
        rm = _manager()
        rm._jwt = _jwt()
        rm._session.post.return_value = Mock(status_code=201, text="")
        rm._session.patch.return_value = Mock(status_code=200, text="")
        rm.post("datapoints", json={"n": 1})
        rm.patch("datapoints/1", json={"n": 2})
        rm._session.post.assert_called_once()
        rm._session.patch.assert_called_once()

    def test_check_error_code_true_on_5xx_false_on_2xx(self):
        rm = _manager()
        err = Mock(status_code=500, text="boom")
        ok = Mock(status_code=200, text="")
        assert rm.check_error_code(err, "https://x", "GET") is True
        assert rm.check_error_code(ok, "https://x", "GET") is False

    def test_check_error_code_can_ignore_status(self):
        rm = _manager()
        resp = Mock(status_code=404, text="missing")
        assert rm.check_error_code(resp, "https://x", "GET", ignore_codes=[404]) is True

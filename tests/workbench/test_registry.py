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

from click.testing import CliRunner

from workbench.challenges.registry import extra_challenge_names, get_spec, shipped_slugs
from workbench.cli import cli


def test_shipped_slugs():
    assert shipped_slugs() == [
        "breaking-rsa",
        "hardening-quantum-proof",
        "mock",
    ]


def test_get_spec_aliases():
    rsa = get_spec("breaking-rsa")
    assert rsa is not None
    assert get_spec("breaking_rsa") is rsa
    assert get_spec("mock-challenge") is get_spec("mock")
    assert get_spec("hqp") is get_spec("hardening-quantum-proof")
    assert get_spec("nope") is None


def test_registry_is_not_listed_as_extra():
    assert "registry" not in extra_challenge_names()


def test_cli_lists_test_and_build():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "test" in result.output
    assert "run" in result.output
    assert "build" in result.output
    assert "challenges" in result.output


def test_cli_run_is_alias_of_test():
    runner = CliRunner()
    test_help = runner.invoke(cli, ["test", "--help"])
    run_help = runner.invoke(cli, ["run", "--help"])
    assert test_help.exit_code == 0
    assert run_help.exit_code == 0
    assert "CHALLENGE" in test_help.output
    assert "CHALLENGE" in run_help.output


def test_cli_unknown_challenge():
    runner = CliRunner()
    result = runner.invoke(cli, ["test", "not-a-challenge", "--solution", "."])
    assert result.exit_code != 0
    assert "Unknown challenge" in result.output

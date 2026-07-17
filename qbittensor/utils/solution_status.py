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

# define enum for solution status
from enum import Enum


class SolutionStatus(Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    SUCCESS = "Success"
    FAILURE = "Failure"


class ValidationFailureReason(str, Enum):
    """Failure reasons."""

    UNKNOWN = "Unknown"
    WALL_TIME_FAILURE = "WallTimeFailure"
    INCORRECT_FAILURE = "IncorrectFailure"
    INVALID_SUBMISSION_FAILURE = "InvalidSubmissionFailure"
    INTERNAL_FAILURE = "InternalFailure"
    ZIP_DOWNLOAD_FAILURE = "ZipDownloadFailure"
    INVALID_ZIP = "InvalidZip"
    MISSING_DOCKERFILE = "MissingDockerfile"
    BUILD_FAILURE = "BuildFailure"
    RUN_FAILURE = "RunFailure"
    IMAGE_VALIDATION_FAILURE = "ImageValidationFailure"
    INVALID_PROGRAM = "InvalidProgram"
    POLICY_VIOLATION = "PolicyViolation"
    UPLOAD_FAILURE = "UploadFailure"

    @property
    def default_message(self) -> str:
        """Human-readable default message for this failure reason."""
        _messages = {
            ValidationFailureReason.UNKNOWN: "Unknown validation failure.",
            ValidationFailureReason.WALL_TIME_FAILURE: "Solution exceeded the allowed runtime (wall time).",
            ValidationFailureReason.INCORRECT_FAILURE: "Solution produced incorrect output.",
            ValidationFailureReason.INVALID_SUBMISSION_FAILURE: "The submission was invalid (corrupt, missing files, or otherwise malformed).",
            ValidationFailureReason.INTERNAL_FAILURE: "An unexpected internal error occurred.",
            ValidationFailureReason.ZIP_DOWNLOAD_FAILURE: "Failed to download the zip from the provided URL.",
            ValidationFailureReason.INVALID_ZIP: "The zip is invalid. It may be corrupted or not a zip at all.",
            ValidationFailureReason.MISSING_DOCKERFILE: "The zip is missing a Dockerfile in the root directory.",
            ValidationFailureReason.BUILD_FAILURE: "Docker failed to build the image from the provided Dockerfile.",
            ValidationFailureReason.RUN_FAILURE: "Docker failed to run the container from the built image.",
            ValidationFailureReason.IMAGE_VALIDATION_FAILURE: "The built Docker image failed validation checks and cannot be run.",
            ValidationFailureReason.INVALID_PROGRAM: "The program provided in the zip is invalid. It may be missing required files, have syntax errors, or fail other validation checks.",
            ValidationFailureReason.POLICY_VIOLATION: "Dockerfile failed security policy checks.",
            ValidationFailureReason.UPLOAD_FAILURE: "Failed to establish upload locations for validator logs or solution output.",
        }
        return _messages.get(self, "Validation failed.")

"""Verified content-addressed evidence stores; cloud writes are conditional and never overwrite."""

import base64
import hashlib
import importlib
import os
import tempfile
from pathlib import Path
from typing import Protocol, cast

from pydantic import Field

from finserve.contracts.deployment import ImmutableModel


class ArtifactRef(ImmutableModel):
    """References bind namespace, digest and byte length; storage paths are derived from the
    digest.
    """

    namespace: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0, strict=True)


class ArtifactStore(Protocol):
    """Metadata registration depends on verified bytes, independent of local versus S3 storage."""

    def put(self, data: bytes) -> ArtifactRef:
        """Publish immutable content and return a verified reference."""
        ...

    def get(self, reference: ArtifactRef) -> bytes:
        """Verify namespace, size and SHA256 before returning content."""
        ...


def verify_content(
    data: bytes, reference: ArtifactRef, namespace: str, maximum_bytes: int
) -> bytes:
    """Never treat filenames, metadata or S3 ETags as proof of the downloaded content."""
    if reference.namespace != namespace or len(data) > maximum_bytes:
        raise ValueError("artifact namespace mismatch or size limit exceeded")
    if len(data) != reference.size_bytes or hashlib.sha256(data).hexdigest() != reference.sha256:
        raise ValueError("artifact checksum or length mismatch")
    return data


class LocalArtifactStore:
    """Atomic publication with POSIX directory fsync; Windows has atomic visibility only."""

    def __init__(self, root: Path, maximum_bytes: int = 64 * 1024 * 1024) -> None:
        """Keep execution evidence outside source and bound memory for this small-artifact store."""
        self.root = root.resolve()
        repository = Path(__file__).resolve().parents[3]
        if self.root == repository or repository in self.root.parents or maximum_bytes < 1:
            raise ValueError("external artifact root and positive byte limit required")
        self.root.mkdir(parents=True, exist_ok=True)
        self.namespace, self.maximum_bytes = self.root.as_uri(), maximum_bytes

    def _path(self, digest: str) -> Path:
        """Validated digests generate all filenames; references cannot inject filesystem paths."""
        return self.root / "sha256" / digest[:2] / digest

    def put(self, data: bytes) -> ArtifactRef:
        """An existing digest is verified, never overwritten, including concurrent publication."""
        if len(data) > self.maximum_bytes:
            raise ValueError("artifact size limit exceeded")
        reference = ArtifactRef(
            namespace=self.namespace, sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data)
        )
        target = self._path(reference.sha256)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
            staging = Path(temporary.name)
        try:
            try:
                os.link(staging, target)
            except FileExistsError:
                pass
            self.get(reference)
        finally:
            staging.unlink()
        if os.name == "posix":
            for directory in (target.parent, target.parent.parent, self.root, self.root.parent):
                descriptor = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        return reference

    def get(self, reference: ArtifactRef) -> bytes:
        """Bound the read before checksum verification, including a corrupted oversized object."""
        with self._path(reference.sha256).open("rb") as artifact:
            data = artifact.read(self.maximum_bytes + 1)
        return verify_content(data, reference, self.namespace, self.maximum_bytes)


class ReadableBody(Protocol):
    """The S3 body is explicitly closed even when verification fails."""

    def read(self, amount: int) -> bytes:
        """Read at most the configured evidence limit plus one overflow byte."""
        ...

    def close(self) -> None:
        """Release the HTTP connection independently of artifact validity."""
        ...


class S3Client(Protocol):
    """The SDK boundary supports botocore contract tests without AWS credentials."""

    def put_object(self, **kwargs: object) -> dict[str, object]:
        """Perform a conditional S3 object write with an explicit SHA256 checksum."""
        ...

    def get_object(self, **kwargs: object) -> dict[str, object]:
        """Return a closeable object body for independent byte verification."""
        ...


class S3ArtifactStore:
    """Content-addressed S3 evidence using If-None-Match, with SDK authentication left to the
    caller.
    """

    def __init__(
        self, bucket: str, prefix: str, client: S3Client, maximum_bytes: int = 64 * 1024 * 1024
    ) -> None:
        """Namespace is fixed at construction; callers cannot redirect individual reads to another
        bucket.
        """
        if not bucket or maximum_bytes < 1:
            raise ValueError("bucket and positive byte limit required")
        self.bucket, self.prefix, self.client = bucket, prefix.strip("/"), client
        self.maximum_bytes = maximum_bytes
        self.namespace = f"s3://{bucket}/{self.prefix}"

    @classmethod
    def from_environment(cls, bucket: str, prefix: str) -> "S3ArtifactStore":
        """Load optional boto3 only for the cloud adapter; never read or serialize credential
        values.
        """
        boto3 = importlib.import_module("boto3")
        return cls(bucket, prefix, cast(S3Client, boto3.client("s3")))

    def _key(self, digest: str) -> str:
        """Digest-only object keys avoid artifact-controlled path selection."""
        return "/".join(part for part in (self.prefix, "sha256", digest[:2], digest) if part)

    def put(self, data: bytes) -> ArtifactRef:
        """On an existing key, independently verify its bytes instead of accepting metadata or
        ETag.
        """
        if len(data) > self.maximum_bytes:
            raise ValueError("artifact size limit exceeded")
        digest = hashlib.sha256(data)
        reference = ArtifactRef(
            namespace=self.namespace, sha256=digest.hexdigest(), size_bytes=len(data)
        )
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=self._key(reference.sha256),
                Body=data,
                IfNoneMatch="*",
                ChecksumSHA256=base64.b64encode(digest.digest()).decode("ascii"),
            )
        except Exception as error:
            response = getattr(error, "response", None)
            details = (
                cast(dict[str, object], response).get("Error")
                if isinstance(response, dict)
                else None
            )
            code = (
                cast(dict[str, object], details).get("Code") if isinstance(details, dict) else None
            )
            if code not in {
                "PreconditionFailed",
                "412",
            }:
                raise
        self.get(reference)
        return reference

    def get(self, reference: ArtifactRef) -> bytes:
        """Checksum downloaded bytes, including successful responses with incorrect object
        metadata.
        """
        if reference.namespace != self.namespace:
            raise ValueError("artifact namespace mismatch")
        response = self.client.get_object(Bucket=self.bucket, Key=self._key(reference.sha256))
        body = cast(ReadableBody, response["Body"])
        try:
            return verify_content(
                body.read(self.maximum_bytes + 1), reference, self.namespace, self.maximum_bytes
            )
        finally:
            body.close()

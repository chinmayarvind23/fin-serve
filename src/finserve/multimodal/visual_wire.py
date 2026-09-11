"""Generated protobuf descriptor with a tiny optional-import bridge instead of untyped stubs.

Regenerate the descriptor with grpc_tools.protoc from contracts/visual.proto; the real
integration test compares descriptor bytes so this public protobuf wire cannot silently drift.
"""

import base64
import importlib
from typing import Any

DESCRIPTOR_BASE64 = (
    "CoYJCgx2aXN1YWwucHJvdG8SEmZpbnNlcnZlLnZpc3VhbC52MSLJAQoPR2VuZXJhdGVSZXF1ZXN0EhUKBmpvYl9p"
    "ZBgBIAEoCVIFam9iSWQSHgoKZ2VuZXJhdGlvbhgCIAEoBFIKZ2VuZXJhdGlvbhIbCglpbWFnZV9yZ2IYAyABKAxS"
    "CGltYWdlUmdiEiUKDm1vZGVsX3JldmlzaW9uGAQgASgJUg1tb2RlbFJldmlzaW9uEhIKBHNlZWQYBSABKA1SBHNl"
    "ZWQSJwoPdGltZW91dF9zZWNvbmRzGAYgASgBUg50aW1lb3V0U2Vjb25kcyIzCghBZG1pdHRlZBInCg93b3JrZXJf"
    "aW5zdGFuY2UYASABKAlSDndvcmtlckluc3RhbmNlIswBCghBcnRpZmFjdBIQCgNwbmcYASABKAxSA3BuZxIWCgZz"
    "aGEyNTYYAiABKAlSBnNoYTI1NhIhCgxtb2RlbF9zaGEyNTYYAyABKAlSC21vZGVsU2hhMjU2EisKEWluaXRpYWxp"
    "emF0aW9uX25zGAQgASgEUhBpbml0aWFsaXphdGlvbk5zEiMKDWdlbmVyYXRpb25fbnMYBSABKARSDGdlbmVyYXRp"
    "b25OcxIhCgxyZW5kZXJpbmdfbnMYBiABKARSC3JlbmRlcmluZ05zIh0KB0ZhaWx1cmUSEgoEY29kZRgBIAEoCVIE"
    "Y29kZSLlAQoLVmlzdWFsRXZlbnQSGgoIc2VxdWVuY2UYASABKA1SCHNlcXVlbmNlEjoKCGFkbWl0dGVkGAIgASgL"
    "MhwuZmluc2VydmUudmlzdWFsLnYxLkFkbWl0dGVkSABSCGFkbWl0dGVkEjoKCGFydGlmYWN0GAMgASgLMhwuZmlu"
    "c2VydmUudmlzdWFsLnYxLkFydGlmYWN0SABSCGFydGlmYWN0EjcKB2ZhaWx1cmUYBCABKAsyGy5maW5zZXJ2ZS52"
    "aXN1YWwudjEuRmFpbHVyZUgAUgdmYWlsdXJlQgkKB3BheWxvYWQicAoOQXR0ZW1wdFJlcXVlc3QSFQoGam9iX2lk"
    "GAEgASgJUgVqb2JJZBIeCgpnZW5lcmF0aW9uGAIgASgEUgpnZW5lcmF0aW9uEicKD3dvcmtlcl9pbnN0YW5jZRgD"
    "IAEoCVIOd29ya2VySW5zdGFuY2UiTgoNQXR0ZW1wdFN0YXR1cxInCg93b3JrZXJfaW5zdGFuY2UYASABKAlSDndv"
    "cmtlckluc3RhbmNlEhQKBXN0YXRlGAIgASgJUgVzdGF0ZTLAAQoMVmlzdWFsV29ya2VyElgKDkdlbmVyYXRlVmlz"
    "dWFsEiMuZmluc2VydmUudmlzdWFsLnYxLkdlbmVyYXRlUmVxdWVzdBofLmZpbnNlcnZlLnZpc3VhbC52MS5WaXN1"
    "YWxFdmVudDABElYKDUNhbmNlbEF0dGVtcHQSIi5maW5zZXJ2ZS52aXN1YWwudjEuQXR0ZW1wdFJlcXVlc3QaIS5m"
    "aW5zZXJ2ZS52aXN1YWwudjEuQXR0ZW1wdFN0YXR1c2IGcHJvdG8z"
)


def message_types() -> dict[str, Any]:
    """Construct standard protobuf classes lazily so normal service imports need no grpc extra."""
    descriptor = importlib.import_module("google.protobuf.descriptor_pb2")
    pool_module = importlib.import_module("google.protobuf.descriptor_pool")
    factory = importlib.import_module("google.protobuf.message_factory")
    document = descriptor.FileDescriptorSet.FromString(base64.b64decode(DESCRIPTOR_BASE64))
    pool = pool_module.DescriptorPool()
    file = pool.AddSerializedFile(document.file[0].SerializeToString())
    return {
        name: factory.GetMessageClass(value) for name, value in file.message_types_by_name.items()
    }

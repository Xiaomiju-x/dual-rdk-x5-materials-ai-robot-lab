# Vision FSD Passive Semantic Bridge

This package is the pure Python core for an isolated, read-only 4K semantic
observer. It accepts arrays supplied by an external inference adapter and
provides:

- a 64x64 robot-centric semantic BEV with calibrated provenance;
- a visibility mask and bounded sparse vector tokens;
- strict freshness, source-quality, calibration, and digest checks;
- a bounded canonical JSON plus zlib cross-X5 payload;
- odometry-warped static and dynamic memories with independent decay;
- occlusion shadow and retained-memory ghost-risk estimates.

It does not acquire camera frames, open sockets, import ROS, publish motion or
TF, access a serial device, or control the validated finals demo. A future
adapter may transport the bytes and publish diagnostics under the independent
vNext namespace; those responsibilities are intentionally outside this core.

The source image contract defaults to at least 3840x2160. Dense probabilities
are quantized to uint8 for transport, visibility is bit-packed, metadata is
canonical ASCII JSON, and SHA-256 authenticates the header and decompressed
arrays. Decoder limits are enforced before allocation and decompression.

# Lyra reconstruction playback bundle

`bundle/` contains the prepared decoder, acoustic matrices and synthetic reference/reconstruction audio used by the local demo. Start it from the repository root with `python -m jaeger_lyra --port 8794`.

The source recording is encoded offline. Live playback decodes the saved representation; it does not generate new sentences. `bundle/checksums.json` records the exact packaged files. See `docs/DATA_PROVENANCE.md` for the Kokoro-derived source/training data and current limitations.

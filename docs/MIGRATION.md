# Migration into JaegerLyra

On November 7, 2026, the audio project was extracted from `AI Visual Playground/local-avatar-demo` into the existing `JaegerLyra` repository.

Moved scope: the full `voice-research/` tree; audio experiment/preparation scripts; audio evidence; historical browser listening rooms; audio license records; optional pretrained model/tool installations; audio environments and caches; and the existing speech/singing adapters. Avatar meshes, rendering, animation, tracking, the avatar server and its main environment remain in their original project.

The old project's moved paths now link to Lyra's files. This preserves the avatar client's current audio dependencies while establishing one local owner for the audio work. The links are outside this repository; a fresh Lyra checkout does not need them. The detailed local inventory is `evidence/migration-v1/local-transfer.json` (not committed because it records machine-specific paths).

The original research directory names are retained so historical experiments, hashes and relative artifact paths stay intelligible. Original historical reports are not rewritten as new results. The demo's versioned files live in `demo/bundle/`; large local archives, caches, environments and external model downloads are ignored. The minimal `.venv` can be recreated from `requirements.txt`.

The initial repository was a Jaeger template. Its unused module/workspace examples were moved to `docs/template-reference/`. The actual README, current-status document, roadmap, demo and CI now describe Lyra rather than the placeholder module.

No claim of a finished TTS/STT system, a working JaegerOS adapter or an end-to-end learned audio model is introduced by this migration.

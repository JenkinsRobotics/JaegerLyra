---
license: apache-2.0
language:
- en
base_model:
- ResembleAI/chatterbox-turbo
pipeline_tag: text-to-speech
library_name: mlx-audio
tags:
- text-to-speech
- speech
- speech generation
- voice cloning
- mlx
- tts
---

# mlx-community/chatterbox-turbo-8bit
This model was converted to MLX format from [`ResembleAI/chatterbox-turbo`](https://huggingface.co/ResembleAI/chatterbox-turbo) using mlx-audio version **0.2.8**.
Refer to the [original model card](https://huggingface.co/ResembleAI/chatterbox-turbo) for more details on the model.
## Use with mlx

```bash
pip install -U mlx-audio
```
Voice Cloning: 
```bash
mlx_audio.tts.generate --model mlx-community/chatterbox-turbo-8bit --text "Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. It's the SkyNet T-800 series and it's got basically everything. Including AI integration with ChatGPT and all that jazz. Would you like me to get some prices for you?" --ref_audio path_to_file.wav --play
```

## Emotion Control

Chatterbox supports expressive event tags that can be inserted directly into your text to add natural vocal expressions:

| Tag | Description |
|-----|-------------|
| `[clear throat]` | Throat clearing sound |
| `[sigh]` | Sighing expression |
| `[shush]` | Shushing sound |
| `[cough]` | Coughing sound |
| `[groan]` | Groaning expression |
| `[sniff]` | Sniffing sound |
| `[gasp]` | Gasping expression |
| `[chuckle]` | Light chuckling |
| `[laugh]` | Laughter |

```bash
mlx_audio.tts.generate --model mlx-community/chatterbox-turbo-8bit --text "[sigh] I can't believe it's Monday again. [groan] But hey, [clear throat] let's make the best of it!" --play
```

Default Voice:
```bash
mlx_audio.tts.generate --model mlx-community/chatterbox-turbo-8bit --text "Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. It's the SkyNet T-800 series and it's got basically everything. Including AI integration with ChatGPT and all that jazz. Would you like me to get some prices for you?" --play
```

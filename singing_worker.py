"""Render a short original vocal performance with local ACE-Step 1.5."""
import json,os,sys
import mlx.core as mx
mx.set_cache_limit(128*1024*1024)
from pathlib import Path
ROOT=Path(__file__).resolve().parent
TOOL=ROOT/'voice-tools/ace-step'
sys.path.insert(0,str(TOOL))
os.environ['ACESTEP_LM_BACKEND']='mlx'
from acestep.handler import AceStepHandler
from acestep.inference import GenerationParams,GenerationConfig,generate_music
import gc,torch

class SongHandler(AceStepHandler):
 def _load_main_model_from_checkpoint(self,**kwargs):
  # Modern macOS supports bfloat16 on MPS. Avoid the upstream float32
  # expansion of all DiT weights on a shared-memory 32 GB machine.
  self.dtype=torch.bfloat16
  return super()._load_main_model_from_checkpoint(**kwargs)
 def _init_mlx_dit(self,compile_model=False):
  ok=super()._init_mlx_dit(compile_model)
  if not ok:raise RuntimeError('Native MLX singing is required on this Mac')
  # The upstream adapter retains two complete diffusion decoders. Only MLX
  # performs diffusion here; keep PyTorch's conditioning modules, free its copy.
  self.model.decoder=torch.nn.Identity()
  import mlx.core as mx
  from mlx.utils import tree_map
  self.mlx_decoder.update(tree_map(lambda x:x.astype(mx.bfloat16) if mx.issubdtype(x.dtype,mx.floating) else x,self.mlx_decoder.parameters()))
  mx.eval(self.mlx_decoder.parameters())
  gc.collect();torch.mps.empty_cache();mx.clear_cache()
  return True

request_path=Path(sys.argv[1]);data=json.loads(request_path.read_text())
handler=SongHandler()
status,ok=handler.initialize_service(project_root=str(TOOL),config_path='acestep-v15-turbo',device='mps',offload_to_cpu=True,offload_dit_to_cpu=True,use_mlx_dit=True)
if not ok:raise RuntimeError(status)
params=GenerationParams(caption=data['caption'],lyrics=data['lyrics'],duration=data.get('duration',15),vocal_language='en',bpm=100,keyscale='C Major',timesignature='4',seed=42,inference_steps=8,thinking=False,use_cot_metas=False,use_cot_caption=False,use_cot_language=False,normalization_db=-2,fade_in_duration=.02,fade_out_duration=.3)
result=generate_music(handler,None,params,GenerationConfig(batch_size=1,audio_format='wav'),save_dir=str(ROOT/'.cache/songs'))
if not result.success:raise RuntimeError(result.error)
request_path.with_suffix('.result.json').write_text(json.dumps({'path':result.audios[0]['path']}))

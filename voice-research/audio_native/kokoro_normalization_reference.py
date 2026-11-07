"""Source-derived normalization ablation for the installed Kokoro decoder.

The normalization equation follows mlx-audio's MIT _InstanceNorm implementation
(licenses/mlx-audio-LICENSE.txt). This isolated-process hook keeps statistics
from a normal source utterance instead of estimating them over an extended
utterance. It is a pretrained-model experiment, not an independent vocoder.
"""
from contextlib import contextmanager
import json
import numpy as np


class NormalizationReference:
    def __init__(self,decoder):
        from mlx_audio.tts.models.kokoro.istftnet import _InstanceNorm
        self.norm_class=_InstanceNorm
        self.names={id(module):name for name,module in decoder.named_modules()
                    if isinstance(module,_InstanceNorm)}
        if not self.names:raise ValueError('No decoder instance-normalization modules')
        self.statistics={}

    @contextmanager
    def apply(self,mode):
        import mlx.core as mx
        if mode not in ('capture','reference'):raise ValueError('Unknown normalization mode')
        if mode=='reference' and not self.statistics:raise ValueError('Capture source statistics first')
        if mode=='capture':self.statistics={}
        original=self.norm_class._apply_instance_norm;counts={}
        def observed(module,x):
            name=self.names.get(id(module))
            if name is None:raise ValueError('Normalization outside the selected decoder')
            if module.affine or module.track_running_stats:raise ValueError('Expected untracked non-affine Kokoro normalization')
            index=counts.get(name,0);counts[name]=index+1
            feature=x.ndim-module._get_no_batch_dim()
            axes=[d for d in range(x.ndim) if d not in (0,feature)]
            if mode=='capture':
                mean=mx.mean(x,axis=axes,keepdims=True);var=mx.var(x,axis=axes,keepdims=True)
                mx.eval(mean,var)
                self.statistics.setdefault(name,[]).append({'mean':mean,'variance':var,
                    'featureAxis':feature,'epsilon':float(module.eps),'sourceShape':tuple(x.shape)})
                return original(module,x)
            entries=self.statistics.get(name,[])
            if index>=len(entries):raise ValueError('Unexpected normalization invocation')
            entry=entries[index]
            if feature!=entry['featureAxis'] or float(module.eps)!=entry['epsilon']:
                raise ValueError('Normalization configuration changed')
            shape=entry['mean'].shape
            if len(shape)!=x.ndim or any(shape[d]!=x.shape[d] for d in (0,feature)):
                raise ValueError('Normalization reference shape changed')
            return (x-entry['mean'])/mx.sqrt(entry['variance']+module.eps)
        self.norm_class._apply_instance_norm=observed
        succeeded=False
        try:
            yield self
            succeeded=True
        finally:
            self.norm_class._apply_instance_norm=original
        if succeeded:
            if not counts:raise ValueError('No normalization invocations')
            if mode=='reference' and counts!={name:len(entries) for name,entries in self.statistics.items()}:
                raise ValueError('Incomplete normalization reference use')

    def save(self,path):
        import mlx.core as mx
        arrays={};entries=[]
        for name,values in self.statistics.items():
            for index,entry in enumerate(values):
                key='stat'+str(len(entries)).zfill(4)
                arrays[key+'_mean']=np.array(entry['mean'].astype(mx.float32))
                arrays[key+'_variance']=np.array(entry['variance'].astype(mx.float32))
                entries.append({'name':name,'invocation':index,'key':key,'featureAxis':entry['featureAxis'],
                    'epsilon':entry['epsilon'],'sourceShape':list(entry['sourceShape']),
                    'dtype':str(entry['mean'].dtype).split('.')[-1]})
        np.savez(path,**arrays,metadata=json.dumps(entries))
        return {'modules':len(self.statistics),'invocations':len(entries),
            'statisticScalars':sum(array.size for array in arrays.values()),
            'scope':'Per-utterance source means and variances; not trained weights or a general voice embedding'}

    @classmethod
    def load(cls,decoder,path):
        import mlx.core as mx
        reference=cls(decoder);known=set(reference.names.values())
        dtypes={'float32':mx.float32,'float16':mx.float16,'bfloat16':mx.bfloat16}
        with np.load(path,allow_pickle=False) as stored:
            entries=json.loads(str(stored['metadata']))
            for entry in entries:
                name=entry['name'];key=entry['key'];dtype=dtypes.get(entry['dtype'])
                values=reference.statistics.setdefault(name,[])
                mean=stored[key+'_mean'];variance=stored[key+'_variance']
                shape=entry['sourceShape'];axis=entry['featureAxis']
                if (name not in known or entry['invocation']!=len(values) or dtype is None
                        or mean.shape!=variance.shape or len(shape)!=mean.ndim or axis not in range(mean.ndim)
                        or not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance<0)
                        or not np.isfinite(entry['epsilon']) or entry['epsilon']<=0
                        or any(mean.shape[d]!=(shape[d] if d in (0,axis) else 1) for d in range(mean.ndim))):
                    raise ValueError('Invalid saved normalization reference')
                values.append({'mean':mx.array(mean,dtype=dtype),'variance':mx.array(variance,dtype=dtype),
                    'featureAxis':axis,'epsilon':entry['epsilon'],'sourceShape':tuple(shape)})
        if not reference.statistics:raise ValueError('Empty normalization reference')
        return reference

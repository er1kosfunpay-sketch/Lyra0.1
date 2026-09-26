"""Colab CUDA and tiny-model sanity checks; never starts full training."""
import argparse
import json
import torch
from lyra.config import LyraConfig
from lyra.model import LyraModel

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/colab.json')
    p.add_argument('--sanity', action='store_true', help='run forward/backward on configs/debug.json')
    p.add_argument('--require-cuda', action='store_true')
    p.add_argument('--instantiate', action='store_true', help='instantiate configured full model on CUDA and verify exact parameter count')
    a = p.parse_args()
    cuda = torch.cuda.is_available()
    if a.require_cuda and not cuda:
        raise SystemExit('CUDA GPU is required for Colab training; select Runtime > Change runtime type > GPU.')
    info = {'torch': torch.__version__, 'cuda_available': cuda, 'cuda_version': torch.version.cuda,
            'device': torch.cuda.get_device_name(0) if cuda else 'cpu'}
    if cuda:
        props = torch.cuda.get_device_properties(0)
        info.update(vram_gib=round(props.total_memory / 1024**3, 2), bf16_supported=torch.cuda.is_bf16_supported())
    c = LyraConfig.from_json(a.config)
    info.update(parameters_from_config=c.parameter_count(), context_length=c.context_length,
                gradient_checkpointing=c.gradient_checkpointing)
    if a.instantiate:
        if not cuda:
            raise SystemExit('--instantiate requires CUDA to avoid allocating the full model on local CPU.')
        model = LyraModel(c).to('cuda')
        actual = model.parameter_count()
        if actual != c.parameter_count():
            raise AssertionError(f'Parameter count mismatch: model={actual}, config={c.parameter_count()}')
        info['instantiated_parameter_count'] = actual
        del model
        torch.cuda.empty_cache()
    print(json.dumps(info, indent=2))
    if a.sanity:
        tiny = LyraConfig.from_json('configs/debug.json')
        m = LyraModel(tiny).to('cuda' if cuda else 'cpu')
        x = torch.randint(0, tiny.vocab_size, (2, min(32, tiny.context_length)))
        y = x.clone()
        loss = m(x, y)['loss']
        loss.backward()
        assert torch.isfinite(loss).item()
        print(json.dumps({'tiny_sanity': 'PASS', 'parameters': m.parameter_count(), 'loss': float(loss)}))

if __name__ == '__main__':
    main()

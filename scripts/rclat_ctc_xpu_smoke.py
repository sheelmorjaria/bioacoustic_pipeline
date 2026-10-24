"""Actual causal GRU/CTC training checks on both XPUs,not only matrix arithmetic."""
import json
from pathlib import Path
import torch
from bioacoustic_pipeline.representation.rclat_ctc import CausalCTC,ctc_objective,encode_text


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    count=torch.xpu.device_count()
    if count < 2:
        raise RuntimeError('Both XPUs must be exposed to this process')
    checks=[]
    for index in range(2):
        device=torch.device(f'xpu:{index}')
        for arm,width in [('mel',64),('physical',12),('combined',76),('pretrained',768)]:
            torch.manual_seed(0)
            model=CausalCTC(arm,width).to(device)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            x=torch.randn(2,50,width,device=device)
            valid=torch.ones_like(x,dtype=torch.bool)
            logits,_=model(x,valid,torch.tensor([40,50]))
            loss=ctc_objective(logits,torch.tensor([40,50]),[encode_text('A B'),encode_text('THE')])
            loss.backward()
            gradient_sum=sum(float(p.grad.detach().abs().sum().cpu()) for p in model.parameters() if p.grad is not None)
            optimizer.step()
            torch.xpu.synchronize(index)
            if not gradient_sum>0 or not all(torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError('Invalid gradients or parameters')
            checks.append({'device':str(device),'device_name':torch.xpu.get_device_name(index),
                'arm':arm,'loss':float(loss.detach().cpu()),'gradient_sum':gradient_sum,
                'ctc_loss_device':'cpu_deterministic','neural_device':str(device),'optimizer_step_passed':True})
    output=Path('outputs/rclat_ctc_xpu_smoke.json')
    if output.exists():
        raise RuntimeError('Device smoke record already exists')
    output.write_text(json.dumps({'torch_version':torch.__version__,'xpu_count':count,
        'strict_deterministic_algorithms':True,'checks':checks,'scientific_model_training':False},indent=2)+'\n')
    print('Both XPUs passed neural backward and optimizer steps with differentiable deterministic CPU CTC for all4frontends.')


if __name__=='__main__':
    main()

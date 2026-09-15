"""Optimistic 1D coast/burn references; never used by the PPO action path.

Closed-form constant-thrust braking with instantaneous spool, no steering or
drag. This restricted trajectory family is NOT a proof of global optimality.
Its minimum-impulse member illustrates gravity losses and a late braking burn.
"""
from pathlib import Path
import json
import math
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def reference(height=18., initial_down_speed=1., impact=.15):
    import torch
    import yaml
    from tvc_env.dynamics.fin_aero import FinAeroModel
    from tvc_env.dynamics.coupled_jet import CoupledJet
    cfg=yaml.safe_load((ROOT/'configs/env/train_512_8s_momentum.yaml').read_text(encoding='utf-8'))
    normals=torch.tensor([[0,-1.,0],[1.,0,0],[0,1.,0],[-1.,0,0]])
    cops=torch.tensor([[[.04,0,.1],[0,.04,.1],[-.04,0,.1],[0,-.04,.1]]])
    model=CoupledJet(cfg['dynamics']['coupled_jet'],FinAeroModel(.002,.262,exhaust_speed=128.),normals)
    jet=model.compute(torch.zeros(1,4),torch.ones(1),torch.tensor([4649.56]),torch.tensor([48.]),cops,torch.zeros(1,4,3),torch.zeros(1,3))
    max_force=48-float(jet.forces[:,:,2].sum())
    mass,g=3.104,9.81
    h=height-.3125
    data=[]
    for force in np.linspace(mass*g*1.001,max_force,500):
        a=force/mass-g
        switch_speed=math.sqrt((2*a*g*h+a*initial_down_speed**2+g*impact**2)/(a+g))
        coast=(switch_speed-initial_down_speed)/g
        burn=(switch_speed-impact)/a
        if coast<0 or burn<=0:continue
        duration=coast+burn
        impulse=force/mass*burn
        # Ideal constant-speed shaft work only: cold spool energy, electrical
        # sag, yaw control, tilt and actuator losses make this optimistic.
        shaft_power=3072*(force/max_force)**1.5
        energy=shaft_power/.88*burn/3600+10*duration/3600
        data.append(dict(thrust_n=float(force),coast_s=coast,burn_s=burn,duration_s=duration,
            energy_wh=energy,delta_v_m_s=impulse,ignition_clearance_m=(switch_speed**2-impact**2)/(2*a),
            ignition_down_speed_m_s=switch_speed))
    return dict(assumptions='Ideal 1D constant-force burn family, instantaneous spool, no steering, no wind, no battery sag; not globally optimal or a deployed controller',
        start_height_m=height,start_down_speed_m_s=initial_down_speed,impact_m_s=impact,
        mass_kg=mass,max_net_thrust_n=max_force,minimum_impulse=min(data,key=lambda r:r['delta_v_m_s']),
        minimum_energy=min(data,key=lambda r:r['energy_wh']),samples=data)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=reference()
    folder=ROOT/'runs/landing_reference';folder.mkdir(parents=True,exist_ok=True)
    (folder/'ideal_1d.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    plt.style.use('dark_background')
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    data=result['samples'];thrust=[r['thrust_n'] for r in data]
    axes[0].plot(thrust,[r['energy_wh'] for r in data],color='#64dfc3')
    axes[0].set(xlabel='Burn thrust after vanes (N)',ylabel='Ideal electrical energy (Wh)',ylim=(2,10))
    axes[1].plot(thrust,[r['delta_v_m_s'] for r in data],color='#6aa9ff')
    axes[1].set(xlabel='Burn thrust after vanes (N)',ylabel='Propulsive impulse / mass (m/s)',ylim=(20,130))
    r=result['minimum_impulse'];t=np.linspace(0,r['duration_s'],300)
    tc=r['coast_s'];v0=result['start_down_speed_m_s'];h0=result['start_height_m']-.3125
    a=r['thrust_n']/result['mass_kg']-9.81
    h=np.where(t<tc,h0-v0*t-.5*9.81*t*t,r['ignition_clearance_m']-r['ignition_down_speed_m_s']*(t-tc)+.5*a*(t-tc)**2)
    axes[2].plot(t,h,color='#64dfc3');axes[2].axvline(tc,color='#ffbf69',ls='--',label='Ideal ignition')
    axes[2].set(xlabel='Time (s)',ylabel='Foot clearance (m)');axes[2].legend()
    fig.suptitle('Optimistic vertical coast/burn references — no steering, spool delay or battery sag',fontsize=12)
    fig.savefig(ROOT/'docs/ideal_landing_reference.png',dpi=150)
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))


if __name__=='__main__':main()

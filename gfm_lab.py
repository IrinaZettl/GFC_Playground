"""
gfm_lab.py - helper functions for the "Grid-following vs. grid-forming" ANDES lab.

The test system is the WSCC / IEEE 9-bus system with TWO generation units:

    * G1  (bus 1): synchronous generator (GENROU + IEEEX1 exciter + TGOV1 governor)
    * IBR (bus 2): inverter-based resource, either
        - grid-following (GFL):  REGCA1 + REECA1 + REPCA1  (WECC generic models)
        - grid-forming  (GFM):  REGF1 (droop) or REGF2 (virtual synchronous machine),
          extended by a current limiter and an optional integral frequency
          controller -> REGF1Lab / REGF2Lab (defined below)

With the integral frequency control switched on, the GFM inverter shifts its own
power set-point until the frequency is back at 50 Hz.

Bus 3 (originally the third machine) is kept as an empty node so the familiar
9-bus topology is preserved.

The functions in this module are intentionally kept short so students can
read them.  The notebook only needs `run_case()` and `plot_results()`.
"""

import logging
import sys
import types

# Do not pass ANDES messages on to the root logger: Google Colab attaches its own
# handler there, which would otherwise print every INFO/WARNING message.
# (Set before `import andes`, which already logs during import.)
logging.getLogger('andes').propagate = False

import andes
import matplotlib.pyplot as plt
import numpy as np
from andes.core import Algeb, ConstService, NumParam, State
from andes.core.block import IntegratorAntiWindup
from andes.core.common import dummify
from andes.core.discrete import HardLimiter
from andes.models.renewable.regf1 import (REGF1Data, REGF1Model, REGF1Primary,
                                          REGFInnerPIModel)
from andes.models.renewable.regf2 import REGF2Data, REGF2Primary

andes.config_logger(stream_level=logging.ERROR)

F_NOM = 50.0     # Hz
S_BASE = 100.0   # MVA, system base
S_IBR = 250.0    # MVA, inverter rating
S_SG = 400.0     # MVA, synchronous generator rating
R_GOV = 0.05     # governor droop of the synchronous generator (p.u. on S_SG)

# --------------------------------------------------------------------------
# Default control parameters (students change these in the notebook)
# --------------------------------------------------------------------------

GFL_DEFAULTS = dict(
    # active power / frequency control (plant controller REPCA1)
    freq_support=True,  # enable P-f droop of the plant controller
    droop=0.05,         # frequency droop R in p.u. (5 % -> 1/R = 20)
    Kpg=0.1,            # P controller proportional gain
    Kig=0.5,            # P controller integral gain
    Tp=0.05,            # active power measurement time constant [s]
    Tg=0.1,             # plant controller output lag [s]
    # reactive power / voltage control (plant controller REPCA1)
    Kp_v=4.0,           # voltage controller proportional gain (>~8 oscillates in a weak grid)
    Ki_v=5.0,           # voltage controller integral gain
    Tfltr=0.02,         # voltage measurement filter [s]
    # converter / measurement
    Tg_conv=0.02,       # converter current time constant (REGCA1.Tg) [s]
    Tf_pll=0.05,        # frequency measurement (PLL-like) time constant [s]
)

GFM_DEFAULTS = dict(
    variant='droop',    # 'droop' (REGF1) or 'vsm' (REGF2, virtual synchronous machine)
    droop=0.05,         # P-f droop m_p in p.u. (5 %)
    q_droop=0.045,      # Q-V droop n_q in p.u. voltage / p.u. reactive power
    Tr=0.02,            # P/Q measurement filter [s] (droop + filter ~ virtual inertia)
    H_vsm=1.0,          # VSM only: virtual inertia constant H [s] (on inverter base)
    D_vsm=0.11,         # VSM only: PLL-based damping factor d_d (weak effect, see notebook)
    xf=0.15,            # coupling (filter) reactance [p.u. on inverter base]
    KPv=3.0,            # voltage controller proportional gain
    KIv=10.0,           # voltage controller integral gain
    KPi=0.5,            # current controller proportional gain
    KIi=20.0,           # current controller integral gain
    Te=0.005,           # converter output voltage time constant [s]
    Pmax=1.2,           # active power limit [p.u. on inverter base]
    current_limit=True, # limit the converter current magnitude to Imax
    Imax=1.2,           # current limit |I| <= Imax [p.u. on inverter base]
    freq_integral=False,  # integral frequency control in the inverter (restores 50 Hz)
    T_I=10.0,           # time constant of the integral frequency control [s]
)

DISTURBANCES = {
    'Load step +50 MW (bus 5)': dict(kind='load', bus=5, p=0.5, q=0.0),
    'Load step +100 MW (bus 5)': dict(kind='load', bus=5, p=1.0, q=0.0),
    '3-ph fault bus 5 (100 ms)': dict(kind='fault', bus=5, tc=0.10, xf=0.02),
    'Line trip 5-7': dict(kind='line', line='Line_2'),
    'Trip of sync. generator G1': dict(kind='gen', dev='GENROU_1'),
}


# --------------------------------------------------------------------------
# Grid-forming models of the lab (REGF1Lab, REGF2Lab)
# --------------------------------------------------------------------------
#
# The ANDES models REGF1 (droop) and REGF2 (VSM) are extended by
#   1) a current limiter and
#   2) an integral frequency controller.
#
# 1) Current limitation
# The ANDES models REGF1 / REGF2 pass the outputs of the voltage controllers
# directly to the current controllers as references, so the inverter current
# is unlimited. A real converter must protect its semiconductors, so the models
# below add a circular current limiter between the two loops:
#
#     I_ul  = output of the voltage PI controllers (d and q axis)
#     I_ref = I_ul                     if |I_ul| <= Imax
#           = I_ul * Imax / |I_ul|     otherwise   (direction is kept)
#
# While the limit is active the voltage controllers would wind up, so their
# integrators track the limited reference (back-calculation anti-windup):
#
#     dx/dt = KIv * e + KIv / KPv * (I_ref - I_ul)
#
# In current limitation the inverter no longer behaves like a voltage source
# behind x_f but like a current source - it can then lose synchronism.
#
# 2) Integral frequency control
# Droop control alone leaves a permanent frequency deviation. The integral
# controller integrates the inverter's own frequency deviation into an extra
# power set-point P_sec, which is added to P_ref in the droop / VSM equation:
#
#     dP_sec/dt = -K_I * (omega - omega_0) / omega_0,    K_I = 1 / (m_p * T_I)
#
# In steady state dP_sec/dt = 0, so the frequency is exactly 50 Hz again.
# P_sec is limited so that P_ref + P_sec stays within [Pmin, Pmax] (anti-windup).
# K_I = 0 switches the integral control off.

class _LabData:
    def __init__(self):
        self.Imax = NumParam(default=999.0, tex_name='I_{max}', unit='p.u.',
                             info='max. converter current magnitude',
                             non_negative=True,
                             power=True,   # p.u. on device base -> system base
                             )
        self.KI = NumParam(default=0.0, tex_name='K_I', non_negative=True,
                           info='gain of the integral frequency control (0 = off)')


class _FrequencyIntegral:
    """Integral frequency controller: extra power set-point Psec_y."""

    def __init__(self):
        self.Psecmax = ConstService(v_str='Pmax - Pref', tex_name='P_{sec,max}')
        self.Psecmin = ConstService(v_str='Pmin - Pref', tex_name='P_{sec,min}')
        self.Psec = IntegratorAntiWindup(u='- KI * dw_y / w0', T=1.0, K=1.0, y0=0.0,
                                         lower=self.Psecmin, upper=self.Psecmax,
                                         tex_name='P_{sec}',
                                         info='power set-point of the integral frequency control')


class _CurrentLimitedVoltageControl:
    """Voltage PI controllers + circular current limiter (replaces REGFOuterPIModel)."""

    def __init__(self):
        self.Kaw = ConstService(v_str='KIv / KPv', tex_name='K_{aw}',
                                info='anti-windup (back-calculation) gain')
        # unlimited outputs of the voltage controllers: PI with anti-windup
        self.xvd = State(info='d-axis voltage controller integrator', tex_name='x_{vd}',
                         v_str='Id0', e_str='KIv * (vref2 - vd) + Kaw * (Idref - Idul)')
        self.xvq = State(info='q-axis voltage controller integrator', tex_name='x_{vq}',
                         v_str='Iq0', e_str='- KIv * vq + Kaw * (Iqref - Iqul)')
        self.Idul = Algeb(info='unlimited d-axis current reference', tex_name='I_{d,ul}',
                          v_str='Id0', e_str='KPv * (vref2 - vd) + xvd - Idul')
        self.Iqul = Algeb(info='unlimited q-axis current reference', tex_name='I_{q,ul}',
                          v_str='Iq0', e_str='- KPv * vq + xvq - Iqul')
        # current magnitude and limiter
        self.Imag = Algeb(info='magnitude of the unlimited current reference', tex_name='|I_{ul}|',
                          v_str='sqrt(Id0**2 + Iq0**2)',
                          e_str='sqrt(Idul**2 + Iqul**2) - Imag')
        self.Ilim = HardLimiter(u=self.Imag, lower=0.0, upper=self.Imax, no_lower=True,
                                allow_adjust=False, tex_name='I_{lim}')
        # limited current references (scaled, direction kept)
        self.Idref = Algeb(info='limited d-axis current reference', tex_name='I_{dref}',
                           v_str='Id0',
                           e_str='Idul * (Ilim_zi + Ilim_zu * Imax / (Imag + 1e-6)) - Idref')
        self.Iqref = Algeb(info='limited q-axis current reference', tex_name='I_{qref}',
                           v_str='Iq0',
                           e_str='Iqul * (Ilim_zi + Ilim_zu * Imax / (Imag + 1e-6)) - Iqref')


def _rewire(block, u):
    """Give an existing ANDES block a new input and rebuild its equations."""
    block.u = dummify(u)
    block.define()


def _rewire_current_controllers(model):
    """Feed the current controllers with the limited references."""
    _rewire(model.PIId, 'Idref - Id')
    _rewire(model.PIIq, 'Iqref - Iq')


class REGF1Lab(REGF1Data, _LabData, REGF1Model, REGF1Primary, _FrequencyIntegral,
               _CurrentLimitedVoltageControl, REGFInnerPIModel):
    """REGF1 (droop GFM) with current limiter and integral frequency control."""

    def __init__(self, system, config):
        REGF1Data.__init__(self)
        _LabData.__init__(self)
        REGF1Model.__init__(self, system, config)
        REGF1Primary.__init__(self)
        _FrequencyIntegral.__init__(self)
        _CurrentLimitedVoltageControl.__init__(self)
        REGFInnerPIModel.__init__(self)
        _rewire_current_controllers(self)
        # droop: omega - omega_0 = m_p * (P_ref + P_sec - P)
        _rewire(self.dw, 'w0 * wdrp * (PIplim_y + Psec_y - Psen_y)')


class REGF2Lab(REGF2Data, _LabData, REGF1Model, REGF2Primary, _FrequencyIntegral,
               _CurrentLimitedVoltageControl, REGFInnerPIModel):
    """REGF2 (VSM GFM) with current limiter and integral frequency control."""

    def __init__(self, system, config):
        REGF2Data.__init__(self)
        _LabData.__init__(self)
        REGF1Model.__init__(self, system, config)
        REGF2Primary.__init__(self)
        _FrequencyIntegral.__init__(self)
        _CurrentLimitedVoltageControl.__init__(self)
        REGFInnerPIModel.__init__(self)
        _rewire_current_controllers(self)
        # swing equation: P_ref is replaced by P_ref + P_sec
        _rewire(self.INTw, '(plldw * dd * wdrp) + wref + wdrp * (PIplim_y + Psec_y - Psen_y)'
                           ' - INTw_y')


def _register_models():
    """
    Make the models known to ANDES: ANDES imports every model listed in
    `andes.models.file_classes` from a module `andes.models.<name>`.
    """
    mod = types.ModuleType('andes.models.gfm_lab_models')
    mod.REGF1Lab, mod.REGF2Lab = REGF1Lab, REGF2Lab
    sys.modules[mod.__name__] = mod
    entry = ('gfm_lab_models', ['REGF1Lab', 'REGF2Lab'])
    fc = andes.models.file_classes
    fc[:] = [e for e in fc if e[0] != entry[0]] + [entry]   # safe on module reload


_register_models()


def _new_system():
    """
    andes.System() with the lab models. ANDES turns the model equations into
    Python code once and caches it in ~/.andes/pycode. For the two new models
    this happens on first use (takes ~1 min); it runs in this process
    because ANDES' parallel code generation cannot see the new models.
    """
    ss = andes.System(no_undill=True)
    ss.undill(autogen_stale=False)
    stale = list(ss.codegen._find_stale_models())
    if stale:
        print(f'Generating code for {", ".join(stale)} (only once, takes ~1 min) ...')
        ss.prepare(quick=True, incremental=True, models=stale, nomp=True)
        # ANDES only reloads the code files it had already imported. Forget the
        # loaded package so later System() calls import it again from disk,
        # including the newly generated files.
        for name in [n for n in sys.modules if n == 'pycode' or n.startswith('pycode.')]:
            del sys.modules[name]
    return ss


# --------------------------------------------------------------------------
# System construction
# --------------------------------------------------------------------------

def _add_network(ss, p_ibr):
    """Buses, lines, transformers, loads and power-flow generators."""
    buses = [  # idx, name, Vn [kV]
        (1, 'Bus 1', 16.5), (2, 'Bus 2', 18.0), (3, 'Bus 3', 13.8),
        (4, 'Bus 4', 230.), (5, 'Bus 5', 230.), (6, 'Bus 6', 230.),
        (7, 'Bus 7', 230.), (8, 'Bus 8', 230.), (9, 'Bus 9', 230.),
    ]
    for idx, name, vn in buses:
        ss.add('Bus', dict(idx=idx, name=name, Vn=vn, v0=1.0, a0=0.0))

    lines = [  # idx, from, to, r, x, b, Vn1, Vn2
        ('Line_0', 5, 4, 0.0100, 0.0680, 0.176, 230, 230),
        ('Line_1', 6, 4, 0.0170, 0.0920, 0.158, 230, 230),
        ('Line_2', 7, 5, 0.0320, 0.1610, 0.306, 230, 230),
        ('Line_3', 9, 6, 0.0390, 0.1738, 0.358, 230, 230),
        ('Line_4', 7, 8, 0.0085, 0.0576, 0.149, 230, 230),
        ('Line_5', 8, 9, 0.0119, 0.1008, 0.209, 230, 230),
        ('Trafo_1', 4, 1, 0.0, 0.0576, 0.0, 230, 16.5),
        ('Trafo_2', 2, 7, 0.0, 0.0625, 0.0, 18.0, 230),
        ('Trafo_3', 9, 3, 0.0, 0.0586, 0.0, 230, 13.8),
    ]
    for idx, b1, b2, r, x, b, vn1, vn2 in lines:
        ss.add('Line', dict(idx=idx, bus1=b1, bus2=b2, r=r, x=x, b=b,
                            Vn1=vn1, Vn2=vn2, fn=F_NOM,
                            trans=int(vn1 != vn2)))

    for idx, bus, p, q in [('Load_5', 5, 1.25, 0.50),
                           ('Load_6', 6, 0.90, 0.30),
                           ('Load_8', 8, 1.00, 0.35)]:
        ss.add('PQ', dict(idx=idx, bus=bus, Vn=230, p0=p, q0=q))

    # G1: synchronous machine, power-flow reference
    ss.add('Slack', dict(idx='G1', bus=1, Sn=S_SG, Vn=16.5, v0=1.04, a0=0.0,
                         p0=1.5, pmax=4.0, pmin=0, qmax=99, qmin=-99))
    # IBR at bus 2 (PV bus in the power flow)
    ss.add('PV', dict(idx='IBR', bus=2, Sn=S_IBR, Vn=18.0, v0=1.025,
                      p0=p_ibr, pmax=2.5, pmin=0, qmax=99, qmin=-99))


def _add_sync_machine(ss, H_sg=4.0, R_gov=R_GOV):
    ss.add('GENROU', dict(idx='GENROU_1', bus=1, gen='G1', Sn=S_SG, Vn=16.5, fn=F_NOM,
                          M=2 * H_sg, D=0.0, ra=0.0, xl=0.15,
                          xd=1.8, xq=1.75, xd1=0.6, xq1=0.8, xd2=0.23, xq2=0.23,
                          Td10=6.5, Tq10=0.2, Td20=0.06, Tq20=0.05,
                          S10=0.09, S12=0.38))
    ss.add('IEEEX1', dict(idx='EXC_1', syn='GENROU_1', TR=0.02, TA=0.02, KA=50,
                          VRMAX=5, VRMIN=-5, KE=1, TE=0.5, KF1=0.06, TF1=1.0,
                          E1=2.8, SE1=0.04, E2=3.73, SE2=0.33))
    ss.add('TGOV1', dict(idx='GOV_1', syn='GENROU_1', R=R_gov, VMAX=1.2, VMIN=0.0,
                         T1=0.5, T2=2.1, T3=7.0, Dt=0.0))


def _add_gfl(ss, p):
    ss.add('REGCA1', dict(idx='REGCA1_1', bus=2, gen='IBR', Sn=S_IBR,
                          Tg=p['Tg_conv'], Rrpwr=999, Brkpt=0.8, Zerox=0.5,
                          Lvpl1=1, Volim=1.2, Lvpnt1=1, Lvpnt0=0.4, Iolim=0,
                          Tfltr=0.02, Khv=0.7, Iqrmax=999, Iqrmin=-999,
                          Accel=0, ra=0, xs=0.25))
    ss.add('REECA1', dict(idx='REECA1_1', reg='REGCA1_1',
                          PFFLAG=0, VFLAG=0, QFLAG=0, PFLAG=0, PQFLAG=0,
                          Vdip=0.8, Vup=1.2, Trv=0.02, dbd1=-0.05, dbd2=0.05,
                          Kqv=2, Tp=0.02, Kqp=1, Kqi=0.1, Kvp=1, Kvi=0.1,
                          Vref0=1.0, Vref1=1.0, Iqh1=999, Iql1=-999,
                          Tiq=0.02, Tpord=0.02, Imax=10,
                          PMAX=1.2, PMIN=0, dPmax=999, dPmin=-999,
                          QMax=999, QMin=-999, VMAX=999, VMIN=-999))
    ss.add('BusFreq', dict(idx='BusFreq_IBR', bus=2, Tf=p['Tf_pll'], Tw=0.02, fn=F_NOM))
    ss.add('REPCA1', dict(idx='REPCA1_1', ree='REECA1_1', line='Trafo_2',
                          busr=2, busf='BusFreq_IBR',
                          # Fflag=1 keeps the plant P control active;
                          # "no frequency support" is realised with zero droop gains
                          VCFlag=1, RefFlag=1, Fflag=1, PLflag=1,
                          Tfltr=p['Tfltr'], Kp=p['Kp_v'], Ki=p['Ki_v'],
                          Tft=0.0, Tfv=0.0, Vfrz=0.7, Kc=0.0,
                          emax=999, emin=-999, dbd1=0, dbd2=0,
                          Qmax=0.8, Qmin=-0.8,
                          Kpg=p['Kpg'], Kig=p['Kig'], Tp=p['Tp'],
                          fdbd1=0, fdbd2=0, femax=999, femin=-999,
                          Pmax=999, Pmin=-999, Tg=p['Tg'],
                          # REPCA1 droop gains are on the system base (100 MVA)
                          Ddn=_gfl_droop_gain(p), Dup=_gfl_droop_gain(p)))


def _gfl_droop_gain(p):
    """REPCA1 droop gain 1/R on system base (0 = no frequency support)."""
    return S_IBR / S_BASE / p['droop'] if p['freq_support'] else 0.0


def _add_gfm(ss, p):
    data = dict(idx='GFM_1', bus=2, gen='IBR', Sn=S_IBR, fn=F_NOM,
                rf=0.0, xf=p['xf'],
                # REGF1/REGF2 apply the droops to P, Q in system p.u. (100 MVA);
                # convert the student's values (inverter base) to system base
                wdrp=p['droop'] * S_BASE / S_IBR,
                Qdrp=p['q_droop'] * S_BASE / S_IBR,
                Tr=p['Tr'], Te=p['Te'],
                KPv=p['KPv'], KIv=p['KIv'], KPi=p['KPi'], KIi=p['KIi'],
                Pmax=p['Pmax'], Pmin=0.0, Qmax=1.0, Qmin=-1.0,
                # The P/Q-limit PI loops of the ANDES implementation would slowly
                # shift the droop set-points; switch them off -> clean droop.
                KPplim=0.0, KIplim=0.0, KPqlim=0.0, KIqlim=0.0,
                # current limit, p.u. on inverter base (999 = practically unlimited)
                Imax=p['Imax'] if p['current_limit'] else 999.0,
                # integral frequency control: K_I = 1 / (m_p * T_I), m_p on system base
                KI=1.0 / (p['droop'] * S_BASE / S_IBR * p['T_I']) if p['freq_integral'] else 0.0)
    if p['variant'] == 'vsm':
        ss.add('REGF2Lab', dict(data, mf=2 * p['H_vsm'] * S_IBR / S_BASE, dd=p['D_vsm']))
    else:
        ss.add('REGF1Lab', data)


def _add_disturbance(ss, dist, t_event):
    if dist['kind'] == 'load':
        # extra load that is switched ON at t_event
        ss.add('PQ', dict(idx='Load_step', bus=dist['bus'], Vn=230,
                          p0=dist['p'], q0=dist['q'], u=0))
        ss.add('Toggle', dict(model='PQ', dev='Load_step', t=t_event))
    elif dist['kind'] == 'fault':
        ss.add('Fault', dict(bus=dist['bus'], tf=t_event, tc=t_event + dist['tc'],
                             xf=dist.get('xf', 1e-4), rf=0))
    elif dist['kind'] == 'line':
        ss.add('Toggle', dict(model='Line', dev=dist['line'], t=t_event))
    elif dist['kind'] == 'gen':
        ss.add('Toggle', dict(model='GENROU', dev=dist['dev'], t=t_event))


def build_system(control='GFM', params=None, disturbance='Load step +50 MW (bus 5)',
                 p_ibr=1.63, t_event=1.0):
    """Create (but do not run) the ANDES system."""
    ss = _new_system()
    _add_network(ss, p_ibr)
    _add_sync_machine(ss)

    if control.upper() == 'GFL':
        p = {**GFL_DEFAULTS, **(params or {})}
        _add_gfl(ss, p)
    elif control.upper() == 'GFM':
        p = {**GFM_DEFAULTS, **(params or {})}
        _add_gfm(ss, p)
    else:
        raise ValueError("control must be 'GFL' or 'GFM'")

    # frequency measurement at every bus (for plotting)
    for b in range(1, 10):
        if not (control.upper() == 'GFL' and b == 2):
            ss.add('BusFreq', dict(idx=f'BusFreq_{b}', bus=b, Tf=0.02, Tw=0.02, fn=F_NOM))

    _add_disturbance(ss, DISTURBANCES[disturbance], t_event)

    # Note: ANDES converts PQ loads to constant impedance for the RMS simulation
    # (default). Constant-power loads would make the fault cases numerically fragile.
    ss.setup()
    return ss


def run_case(control='GFM', params=None, disturbance='Load step +50 MW (bus 5)',
             p_ibr=1.63, t_end=20.0, t_event=1.0, verbose=False,
             h_fine=1 / 120, h_coarse=None, t_fine=5.0):
    """
    Build the system, solve the power flow and run the RMS simulation.

    Time step: h_fine until t_fine seconds after the event (fast inverter and
    network dynamics), then h_coarse (slow governor / integral control dynamics).
    This keeps long simulations reasonably fast. h_coarse=None picks
    0.1 s for GFM and 1/30 s for GFL (the GFL models converge poorly with larger steps).
    """
    if h_coarse is None:
        h_coarse = 1 / 30 if control.upper() == 'GFL' else 0.1
    ss = build_system(control, params, disturbance, p_ibr, t_event)
    ss.PFlow.run()
    if not ss.PFlow.converged:
        raise RuntimeError('Power flow did not converge.')
    if control.upper() == 'GFM':
        _check_initial_current(ss, {**GFM_DEFAULTS, **(params or {})})
    ss.TDS.config.no_tqdm = True
    ss._lab_control = control.upper()
    ss._lab_t_event = t_event

    # stage 1: fine time step around the disturbance
    ss.TDS.config.tstep = h_fine
    ss.TDS.config.tf = min(t_event + t_fine, t_end)
    ss.TDS.run()
    # stage 2: coarser time step for the slow dynamics
    if ss.dae.t < t_end and not ss.TDS.busted:
        ss.TDS.config.tstep = h_coarse
        ss.TDS.config.tf = t_end
        ss.TDS.run()

    if verbose:
        print(f"TDS finished OK: {not ss.TDS.busted}")
    return ss


def _check_initial_current(ss, p):
    """The power-flow operating point must not already exceed the current limit."""
    if not p['current_limit']:
        return
    i0 = np.hypot(ss.PV.p.v[0], ss.PV.q.v[0]) * S_BASE / S_IBR / ss.Bus.v.v[1]
    if i0 > p['Imax']:
        raise ValueError(f'The initial inverter current ({i0:.2f} p.u.) exceeds the current '
                         f'limit Imax = {p["Imax"]:.2f} p.u. Reduce the dispatch or raise Imax.')


# --------------------------------------------------------------------------
# Single-line diagram
# --------------------------------------------------------------------------

def draw_grid(ax=None):
    """Draw a simple single-line diagram of the test system."""
    pos = {1: (0, 0), 4: (0, 1), 5: (-1, 2), 6: (1, 2), 7: (-1, 3), 9: (1, 3),
           8: (0, 3.6), 2: (-2.2, 3), 3: (2.2, 3)}
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5.5))
    branches = [(5, 4), (6, 4), (7, 5), (9, 6), (7, 8), (8, 9), (4, 1), (2, 7), (9, 3)]
    for a, b in branches:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        ax.plot([x1, x2], [y1, y2], color='0.35', lw=2, zorder=1)
    for b, (x, y) in pos.items():
        ax.plot([x - 0.18, x + 0.18], [y, y], color='k', lw=5, zorder=2)
        ax.text(x + 0.22, y + 0.08, f'{b}', fontsize=11, weight='bold')
    # generators
    x, y = pos[1]
    ax.add_patch(plt.Circle((x, y - 0.4), 0.18, fc='#9ecae1', ec='k', zorder=3))
    ax.text(x, y - 0.4, 'SG', ha='center', va='center', fontsize=9, weight='bold')
    ax.text(x + 0.3, y - 0.45, 'G1: synchronous gen.\n400 MVA (GENROU+AVR+Gov)', fontsize=8,
            va='center')
    x, y = pos[2]
    ax.add_patch(plt.Rectangle((x - 0.6, y - 0.18), 0.36, 0.36, fc='#fdae6b', ec='k', zorder=3))
    ax.text(x - 0.42, y, '=\n~', ha='center', va='center', fontsize=9, weight='bold')
    ax.plot([x - 0.24, x - 0.18], [y, y], 'k', lw=2)
    ax.text(x - 0.42, y - 0.42, 'Inverter (GFL / GFM)\n250 MVA', fontsize=8, ha='center',
            va='top')
    ax.text(pos[3][0], pos[3][1] - 0.35, '(no generator)', fontsize=8, ha='center',
            color='0.4')
    # loads
    for b, txt in [(5, '125 MW'), (6, '90 MW'), (8, '100 MW')]:
        x, y = pos[b]
        ax.annotate('', xy=(x, y - 0.4), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', lw=1.8, color='C3'))
        ax.text(x + 0.05, y - 0.35, txt, fontsize=8, color='C3')
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-0.8, 4.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Modified IEEE 9-bus test system ({F_NOM:.0f} Hz)')
    return ax


# --------------------------------------------------------------------------
# Result extraction & plotting
# --------------------------------------------------------------------------

def _ibr_model(ss):
    """Return the ANDES model instance that represents the inverter."""
    for name in ('REGF1Lab', 'REGF2Lab', 'REGCA1'):
        m = getattr(ss, name)
        if m.n > 0:
            return m
    raise RuntimeError('No inverter model found in system.')


def get_results(ss):
    """Return a dict of time series (numpy arrays) from a finished simulation."""
    ts = ss.dae.ts
    t = ts.t
    bf = ss.BusFreq
    f_idx = {b: i for i, b in enumerate(bf.bus.v)}
    freq = {b: F_NOM * ts.y[:, bf.f.a[i]] for b, i in f_idx.items()}
    volt = {b: ts.y[:, ss.Bus.v.a[i]] for i, b in enumerate(ss.Bus.idx.v)}

    m = _ibr_model(ss)
    p_ibr = ts.y[:, m.Pe.a[0]] * S_BASE
    q_ibr = ts.y[:, m.Qe.a[0]] * S_BASE

    g = ss.GENROU
    p_sg = ts.y[:, g.Pe.a[0]] * S_BASE
    q_sg = ts.y[:, g.Qe.a[0]] * S_BASE
    w_sg = F_NOM * ts.x[:, g.omega.a[0]]

    # inverter current magnitude |I| = |S| / V in p.u. on the inverter rating
    i_ibr = np.hypot(p_ibr, q_ibr) / S_IBR / volt[2]

    # GFM with integral frequency control: power set-point shift P_sec [MW]
    p_sec = None
    if hasattr(m, 'Psec_y') and m.KI.v[0] > 0:
        p_sec = ts.x[:, m.Psec_y.a[0]] * S_BASE

    return dict(t=t, freq=freq, volt=volt, p_ibr=p_ibr, q_ibr=q_ibr, i_ibr=i_ibr,
                p_sec=p_sec, p_sg=p_sg, q_sg=q_sg, f_sg=w_sg)


def plot_results(results, labels=None, buses=(2, 5, 7), title=None):
    """
    Plot frequency, voltage and power for one or several simulations.

    results : ANDES system or list of systems (from run_case)
    labels  : list of legend labels (one per system)
    """
    if not isinstance(results, (list, tuple)):
        results = [results]
    if labels is None:
        labels = [r._lab_control for r in results]

    fig, ax = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    styles = ['-', '--', ':', '-.']

    for k, (ss, lab) in enumerate(zip(results, labels)):
        r = get_results(ss)
        ls = styles[k % len(styles)]
        multi = len(results) > 1
        for j, b in enumerate(buses):
            c = f'C{j}'
            name = f'Bus {b}' + (f' ({lab})' if multi else '')
            ax[0, 0].plot(r['t'], r['freq'][b], ls, color=c, label=name)
            ax[0, 1].plot(r['t'], r['volt'][b], ls, color=c, label=name)
        sfx = f' ({lab})' if multi else ''
        ax[1, 0].plot(r['t'], r['p_ibr'], ls, color='C3', label='P inverter' + sfx)
        ax[1, 0].plot(r['t'], r['p_sg'], ls, color='C4', label='P sync. gen.' + sfx)
        if r['p_sec'] is not None:   # set-point shifted by the integral control
            ax[1, 0].plot(r['t'], r['p_ibr'][0] + r['p_sec'], ls, color='C3', lw=0.8, alpha=0.6,
                          label='P set-point inverter (P_ref + P_sec)' + sfx)
        ax[1, 1].plot(r['t'], r['q_ibr'], ls, color='C3', label='Q inverter' + sfx)
        ax[1, 1].plot(r['t'], r['q_sg'], ls, color='C4', label='Q sync. gen.' + sfx)

    ax[0, 0].set_ylabel('Frequency [Hz]')
    ax[0, 1].set_ylabel('Voltage [p.u.]')
    ax[1, 0].set_ylabel('Active power [MW]')
    ax[1, 1].set_ylabel('Reactive power [Mvar]')
    for a in ax.flat:
        a.grid(alpha=0.3)
        a.legend(fontsize=8, loc='best')
    for a in ax[1]:
        a.set_xlabel('Time [s]')
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def metrics(ss, bus=5, window=0.5):
    """
    Key figures of merit at one bus.

    RoCoF is evaluated over a sliding window (default 500 ms) as in grid codes,
    which filters out the numerical spikes right at the switching instant.
    """
    r = get_results(ss)
    t, f = r['t'], r['freq'][bus]
    t0 = ss._lab_t_event
    after = t >= t0
    dev = f[after] - F_NOM
    i_ext = np.argmax(np.abs(dev))

    # sliding-window RoCoF within the first 2 s after the event
    t_grid = np.arange(t0, min(t0 + 2.0, t[-1]) - window, 0.01)
    f_int = np.interp(t_grid, t, f), np.interp(t_grid + window, t, f)
    rocof = (f_int[1] - f_int[0]) / window if len(t_grid) else np.array([np.nan])

    return dict(f_extreme=f[after][i_ext], t_extreme=t[after][i_ext],
                rocof=rocof[np.argmax(np.abs(rocof))],
                f_end=f[-1], v_min=r['volt'][bus][after].min(),
                p_ibr_0=r['p_ibr'][0], p_ibr_end=r['p_ibr'][-1],
                i_ibr_max=r['i_ibr'][after].max(),
                p_sec_end=None if r['p_sec'] is None else r['p_sec'][-1],
                finished=not ss.TDS.busted)


def summary(ss, bus=5, label=None):
    """Print key figures of merit: frequency nadir/peak, RoCoF, steady-state values."""
    m = metrics(ss, bus)
    label = label or ss._lab_control
    flag = '' if m['finished'] else '   (simulation diverged!)'
    print(f'{label:38s} | bus {bus}: f_nadir/peak = {m["f_extreme"]:.3f} Hz '
          f'@ {m["t_extreme"]:.2f} s | RoCoF(500 ms) = {m["rocof"]:+.3f} Hz/s | '
          f'f_end = {m["f_end"]:.3f} Hz | V_min = {m["v_min"]:.3f} pu | '
          f'P_inv {m["p_ibr_0"]:.0f} -> {m["p_ibr_end"]:.0f} MW | '
          f'I_inv,max = {m["i_ibr_max"]:.2f} pu{flag}')
    if m['p_sec_end'] is not None:
        print(f'{"":38s} |   integral frequency control: set-point shift of the inverter '
              f'P_sec = {m["p_sec_end"]:+.1f} MW')


# --------------------------------------------------------------------------
# Interactive control panel (ipywidgets)
# --------------------------------------------------------------------------

def make_gui():
    """Build the interactive lab panel for the Jupyter notebook."""
    import ipywidgets as w
    from IPython.display import display, clear_output

    style = {'description_width': '190px'}

    def lay():
        # one Layout per widget: visibility is toggled via layout.display,
        # a shared Layout would hide/show all widgets at once
        return w.Layout(width='470px')

    def fslider(desc, val, lo, hi, step, fmt='.3f'):
        return w.FloatSlider(value=val, min=lo, max=hi, step=step, description=desc,
                             readout_format=fmt, continuous_update=False,
                             style=style, layout=lay())

    # --- general settings ---
    control = w.ToggleButtons(options=[('Grid following (GFL)', 'GFL'),
                                       ('Grid forming - droop', 'GFM-droop'),
                                       ('Grid forming - VSM', 'GFM-vsm')],
                              value='GFM-droop', description='Inverter control:',
                              style={'description_width': '120px',
                                     'button_width': '170px'})
    disturbance = w.Dropdown(options=list(DISTURBANCES), description='Disturbance:',
                             style=style, layout=lay())
    p_ibr = fslider('Inverter dispatch P [MW]', 163, 20, 240, 1, '.0f')
    t_end = fslider('Simulation time [s]', 15, 3, 180, 1, '.0f')
    keep = w.Checkbox(value=False, description='Keep previous runs for comparison')

    # --- GFL parameters ---
    g = GFL_DEFAULTS
    gfl = dict(
        freq_support=w.Checkbox(value=g['freq_support'], description='Frequency support (P-f droop)'),
        droop=fslider('Droop R [p.u.]', g['droop'], 0.01, 0.20, 0.005),
        Kpg=fslider('P-ctrl Kp', g['Kpg'], 0.0, 2.0, 0.05),
        Kig=fslider('P-ctrl Ki [1/s]', g['Kig'], 0.05, 5.0, 0.05),
        Tp=fslider('P meas. time const. Tp [s]', g['Tp'], 0.01, 0.5, 0.01),
        Tg=fslider('Plant ctrl lag Tg [s]', g['Tg'], 0.01, 1.0, 0.01),
        Kp_v=fslider('V-ctrl Kp', g['Kp_v'], 0.0, 30.0, 0.5, '.1f'),
        Ki_v=fslider('V-ctrl Ki [1/s]', g['Ki_v'], 0.0, 30.0, 0.5, '.1f'),
        Tf_pll=fslider('Freq. meas. (PLL) T [s]', g['Tf_pll'], 0.01, 0.5, 0.01),
        Tg_conv=fslider('Converter current T [s]', g['Tg_conv'], 0.01, 0.2, 0.005),
    )
    # --- GFM parameters ---
    g = GFM_DEFAULTS
    gfm = dict(
        droop=fslider('P-f droop m_p [p.u.]', g['droop'], 0.01, 0.20, 0.005),
        q_droop=fslider('Q-V droop n_q [p.u.]', g['q_droop'], 0.0, 0.15, 0.005),
        Tr=fslider('P/Q meas. filter Tr [s]', g['Tr'], 0.005, 1.0, 0.005),
        H_vsm=fslider('VSM inertia H [s]', g['H_vsm'], 0.1, 10.0, 0.1, '.1f'),
        D_vsm=fslider('VSM PLL damping d_d', g['D_vsm'], -5.0, 5.0, 0.01, '.2f'),
        xf=fslider('Coupling reactance xf [p.u.]', g['xf'], 0.05, 0.4, 0.01, '.2f'),
        KPv=fslider('V-ctrl Kp', g['KPv'], 0.5, 10.0, 0.1, '.1f'),
        KIv=fslider('V-ctrl Ki [1/s]', g['KIv'], 1.0, 50.0, 1.0, '.0f'),
        KPi=fslider('I-ctrl Kp', g['KPi'], 0.1, 2.0, 0.05, '.2f'),
        KIi=fslider('I-ctrl Ki [1/s]', g['KIi'], 5.0, 100.0, 1.0, '.0f'),
        current_limit=w.Checkbox(value=g['current_limit'], description='Current limitation'),
        Imax=fslider('Current limit Imax [p.u.]', g['Imax'], 0.8, 2.0, 0.05, '.2f'),
        freq_integral=w.Checkbox(value=g['freq_integral'],
                                 description='Integral frequency control (restores 50 Hz)'),
        T_I=fslider('Integral time constant T_I [s]', g['T_I'], 2.0, 120.0, 1.0, '.0f'),
    )
    gfm['current_limit'].observe(
        lambda change: setattr(gfm['Imax'], 'disabled', not change['new']), 'value')

    def integral_toggled(change):
        gfm['T_I'].disabled = not change['new']
        if change['new'] and t_end.value < 90:
            t_end.value = 90    # the integral control is slow -> simulate longer

    gfm['freq_integral'].observe(integral_toggled, 'value')
    integral_toggled({'new': gfm['freq_integral'].value})

    gfl_box = w.VBox([w.HTML('<b>Grid-following parameters</b> (REGCA1 / REECA1 / REPCA1)')]
                     + list(gfl.values()))
    gfm_box = w.VBox([w.HTML('<b>Grid-forming parameters</b> (REGF1 droop / REGF2 VSM)')]
                     + list(gfm.values()))

    run_btn = w.Button(description='Run RMS simulation', button_style='success',
                       icon='play', layout=w.Layout(width='220px'))
    reset_btn = w.Button(description='Reset parameters', icon='undo',
                         layout=w.Layout(width='180px'))
    status = w.HTML()
    out = w.Output()
    history = []

    def refresh_visibility(*_):
        is_gfl = control.value == 'GFL'
        gfl_box.layout.display = None if is_gfl else 'none'
        gfm_box.layout.display = 'none' if is_gfl else None
        vsm = control.value == 'GFM-vsm'
        gfm['H_vsm'].layout.display = None if vsm else 'none'
        gfm['D_vsm'].layout.display = None if vsm else 'none'
        gfm['Tr'].layout.display = 'none' if vsm else None

    def reset(_):
        for k, wid in gfl.items():
            wid.value = GFL_DEFAULTS[k]
        for k, wid in gfm.items():
            wid.value = GFM_DEFAULTS[k]

    def run(_):
        if control.value == 'GFL':
            ctrl, params = 'GFL', {k: v.value for k, v in gfl.items()}
            label = f'GFL R={params["droop"]:.3f}' + ('' if params['freq_support'] else ' (no f-supp.)')
        else:
            variant = control.value.split('-')[1]
            ctrl = 'GFM'
            params = {k: v.value for k, v in gfm.items()}
            params['variant'] = variant
            label = f'GFM-{variant} m_p={params["droop"]:.3f}'
            if variant == 'vsm':
                label += f' H={params["H_vsm"]:.1f}'
            if params['freq_integral']:
                label += f' T_I={params["T_I"]:.0f}s'
        label += f', {p_ibr.value:.0f} MW'

        status.value = ('<i>Running power flow and RMS simulation ...'
                        + (' (long simulations take up to ~1 min)' if t_end.value > 40 else '')
                        + '</i>')
        run_btn.disabled = True
        try:
            ss = run_case(ctrl, params=params, disturbance=disturbance.value,
                          p_ibr=p_ibr.value / S_BASE, t_end=t_end.value)
        except Exception as exc:  # show the error instead of crashing the GUI
            status.value = f'<b style="color:#c00">Error:</b> {exc}'
            run_btn.disabled = False
            return
        run_btn.disabled = False
        if ss.TDS.busted:
            status.value = (f'<b style="color:#c00">Simulation diverged at t = {ss.dae.t:.2f} s</b> '
                            '- the system (or the numerical solution) became unstable. '
                            'Results up to that point are shown.')
        else:
            status.value = '<b style="color:#080">Simulation finished.</b>'

        if not keep.value:
            history.clear()
        history.append((ss, label))
        with out:
            clear_output(wait=True)
            plot_results([h[0] for h in history], [h[1] for h in history],
                         title=disturbance.value)
            for s, lab in history:
                summary(s, label=lab)

    control.observe(refresh_visibility, 'value')
    run_btn.on_click(run)
    reset_btn.on_click(reset)
    refresh_visibility()

    general = w.VBox([w.HTML('<b>Scenario</b>'), disturbance, p_ibr, t_end, keep])
    panel = w.VBox([control,
                    w.HBox([general, gfl_box, gfm_box]),
                    w.HBox([run_btn, reset_btn, status]),
                    out])
    display(panel)

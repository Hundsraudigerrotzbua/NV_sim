import numpy as np
from pylab import *
from qutip import *
from functools import wraps
from time import time
from dataclasses import dataclass, replace, field
import logging


import matplotlib

matplotlib.use('TkAgg')


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % (f.__name__, te - ts))
        return result

    return wrap


@dataclass
class system:
    """
    TODO: Update for general system
    """
    external_magnetic_fields: list = field(default_factory=list)
    Sx: Qobj = operators.spin_Jx(1)
    Sy: Qobj = operators.spin_Jy(1)
    Sz: Qobj = operators.spin_Jz(1)
    collapsing: bool = True
    D: float = 2.87e9  # Hz
    gamma: float = 2.8e6  # Hz / Gauss
    B0: float = 15  # Gauss
    B1: float = 0  # Gauss
    T1: float = 10e-6  # seconds
    T2: float = 1e-6  # seconds
    phase: float = 0  # degree
    detuning: float = 0  # Hz (omega)
    phase_offset: float = 0  # degree
    transition: int = 1  # +1 or -1
    omega: float = 2.87e9

    def __post_init__(self):
        self.H = []
        if self.collapsing:
            self.dephasing = 1 / np.sqrt(self.T2) * self.Sz
            self.decay_x = 1 / np.sqrt(self.T1) * self.Sx
            self.decay_y = 1 / np.sqrt(self.T1) * self.Sy
        else:
            self.dephasing = qeye(3)
            self.decay_x = qeye(3)
            self.decay_y = qeye(3)
        self.collapse_operator = [self.dephasing, self.decay_x, self.decay_y]


        self.B_field = np.array([
            lambda t: (0 + self.B1 * np.cos(
                (self.omega + self.detuning) * t + np.deg2rad(self.phase + self.phase_offset))),
            lambda t: 0,#(0 + self.B1 * np.sin(
#                (self.omega + self.detuning) * t + np.deg2rad(self.phase + self.phase_offset))),
            lambda t: self.B0,
        ])

        self.create_hamiltonian(self.ZFS(self.D))
        self.create_hamiltonian(self.magnetic_field_interaction(self.B_field))
        for x in self.external_magnetic_fields: self.create_hamiltonian(self.magnetic_field_interaction(x))

        if self.B1 > 0:
            self.pi_time = np.pi / (self.gamma * self.B1)
        else:
            self.pi_time = np.Inf

    def update_system(self, **kwargs):
        self = replace(self, **kwargs)
        self.__post_init__()
        return self

    def create_hamiltonian(self, field):
        self.H += field

    def add_magnetic_field(self, fields):
        return self.update_system(external_magnetic_fields=fields)

    def ZFS(self, D):
        return [self.Sz ** 2 * D]  # Double, to make merging with hamiltonian intuitive

    def magnetic_field_interaction(self, vector):
        def func_x(t, args):
            return self.gamma * vector[0](t)

        def func_y(t, args):
            return self.gamma * vector[1](t)

        def func_z(t, args):
            return self.gamma * vector[2](t)

        return [[self.Sx, func_x], [self.Sy, func_y], [self.Sz, func_z]]


class pulse():
    @timing
    def __init__(self, hamiltonian, psi0, t):
        self.times = t
        self.states = mesolve(hamiltonian.H, psi0, t,
                              c_ops=hamiltonian.collapse_operator,
                              #args=hamiltonian.args,
                              ).states


class eigenvalue():
    @timing
    def __init__(self, hamiltonian, psi0, t, eops):
        self.times = t
        self.states = mesolve(hamiltonian.H, psi0, t,
                              c_ops=hamiltonian.collapse_operator,
                              e_ops=eops
                              #args=hamiltonian.args,
                              )


class three_level_sim():
    def __init__(self, n_steps=100, collapse=True, **kwargs):
        self.k1, self.k0, self.km1 = three_level_basis()
        self.current_state = ket2dm(self.k0)
        self.sequence = []
        self.n_steps = n_steps
        self.t_vec = np.array([])
        self.system = system()  # NV_hamiltonian(**kwargs)
        self.system.collapsing = collapse

    def eigenvals(self, t_arr):
        return eigenvalue(self.system, self.current_state, t_arr, [self.k1 * self.k1.dag(), self.k0 * self.k0.dag()])

    def evolve(self, duration, **kwargs):
        t_arr = np.linspace(0, duration, self.n_steps)
        if len(self.t_vec) > 0:
            self.t_vec = np.concatenate((self.t_vec, self.t_vec[-1] + t_arr))
        else:
            self.t_vec = t_arr
        p = pulse(self.system, self.current_state, t_arr)
        phase_offset = self.system.phase_offset + 360 * self.system.omega / (2 * np.pi) * duration
        self.system = self.system.update_system(phase_offset=phase_offset, **kwargs)
        self.sequence.append(p)
        self.current_state = p.states[-1]

    def get_states(self):
        return [x for i in self.sequence for x in i.states]

    def pi_pulse(self, **kwargs):
        self.system = self.system.update_system(**kwargs)
        self.evolve(self.system.pi_time*1.5)

    def pi_half_pulse(self, **kwargs):
        self.system = self.system.update_system(**kwargs)
        self.evolve(self.system.pi_time / 2)

    def rabi(self, tau, driving_strength, **kwargs):
        self.system = self.system.update_system(B1=driving_strength, **kwargs)
        self.evolve(tau, B1=driving_strength, **kwargs)

    def ramsey(self, tau, driving_strength, **kwargs):
        self.pi_half_pulse(B1=driving_strength, **kwargs)
        self.evolve(tau, B1=0, **kwargs)
        self.pi_half_pulse(B1=driving_strength, **kwargs)

    def hahn(self, tau, driving_strength, **kwargs):
        self.pi_half_pulse(B1=driving_strength, **kwargs)
        self.evolve(tau / 2, B1=0, **kwargs)
        self.pi_pulse(B1=driving_strength, **kwargs)
        self.evolve(tau / 2, B1=0, **kwargs)
        self.pi_half_pulse(B1=driving_strength, phase=180, **kwargs)

    def pulsed_odmr(self, driving_strength, freq_shift, **kwargs):
        self.pi_pulse(detuning=freq_shift, B1=driving_strength, **kwargs)


def perform_ramsey(n):
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))

    t_arr = np.linspace(1e-9, 800e-9, 50)

    res2 = []

    BzSignalAmp = 10
    BzSignalOmega = np.pi / 2
    BzSignalPhase = 0

    BzNoiseAmp = 0
    BzNoiseOmega = 8 * np.pi * 3e6
    BzNoiseOffset = 0

    noise2 = np.array([
        lambda t: 0,
        lambda t: 0,
        lambda t: 10 * np.cos(0 * t + 0)
    ])

    N = 1
    total_res = []

    for _N in range(N):
        res = []
        BzNoisePhase = np.random.uniform(0, 2 * np.pi)
        noise = np.array([
            lambda t: 0,
            lambda t: 0,
            lambda t: BzSignalAmp + BzNoiseAmp * np.sin(BzNoiseOmega * t + BzNoisePhase) + BzNoiseOffset
        ])
        for idx, t_ev in enumerate(t_arr):
            print(f'Point {idx + 1}/{len(t_arr)}')
            B = three_level_sim(n_steps=n, collapse=True)
            B.system = B.system.add_magnetic_field(
                [noise])  # Has to be done, since dataclass.replace needs to update the class
            B.ramsey(t_ev, driving_strength=30)

            res.append(np.real(expect(B.km1 * B.km1.dag(), B.get_states()))[-1] -
                       np.real(expect(B.k0 * B.k0.dag(), B.get_states()))[-1])

            ax.plot(B.t_vec, np.sqrt(2) * np.real(expect(B.system.Sx, B.get_states())), '--', label=f'Expect of Sx')
            ax.plot(B.t_vec, np.sqrt(2) * np.real(expect(B.system.Sy, B.get_states())), label=f'Expect of Sy')
            ax.plot(B.t_vec, np.real(expect(B.system.Sz ** 2, B.get_states())), label=f'Expect of Sz')
            ax.plot(B.t_vec, np.real(expect(B.k0 * B.k0.dag(), B.get_states())), label=f'Expect of k0')
            ax.plot(B.t_vec, np.real(expect(B.k1 * B.k1.dag(), B.get_states())), label=f'Expect of k1')
            ax.plot(B.t_vec, np.real(expect(B.km1 * B.km1.dag(), B.get_states())), label=f'Expect of km1')
            ax.plot(B.t_vec, np.sin(BzNoiseOmega * B.t_vec), label=f'Signal')
            ax.plot([], [], label=f'{t_ev}')
            plt.grid()
            plt.legend()
            plt.pause(1 / 24)
            plt.cla()
        total_res.append(res)

    plt.figure()
    for X in total_res:
        plt.plot(t_arr, np.array(X),
                 label=f'Pi_time: {B.system.pi_time * 1e9:.02f} ns')
        # plt.plot(t_arr, 1 - np.array(res2))
    plt.ylim((-1, 1))
    plt.legend()
    plt.show()

def perform_rabi(n):
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))

    t_arr = np.linspace(1e-9, 1500e-9, 50)

    res2 = []

    BzSignalAmp = 0
    BzSignalOmega = np.pi / 2
    BzSignalPhase = 0

    BzNoiseAmp = 0
    BzNoiseOmega = 8 * np.pi * 3e6
    BzNoiseOffset = 0

    noise2 = np.array([
        lambda t: 0,
        lambda t: 0,
        lambda t: 10 * np.cos(0 * t + 0)
    ])

    N = 1
    total_res = []

    for _N in range(N):
        res = []
        BzNoisePhase = np.random.uniform(0, 2 * np.pi)
        noise = np.array([
            lambda t: 0,
            lambda t: 0,
            lambda t: BzSignalAmp + BzNoiseAmp * np.sin(BzNoiseOmega * t + BzNoisePhase) + BzNoiseOffset
        ])
        for idx, t_ev in enumerate(t_arr):
            print(f'Point {idx + 1}/{len(t_arr)}')
            B = three_level_sim(n_steps=n, collapse=True)
            B.system = B.system.add_magnetic_field(
                [noise])  # Has to be done, since dataclass.replace needs to update the class
            B.rabi(t_ev, driving_strength=3, detuning=-B.system.B0*2.8e6)

            #res.append(np.real(expect(B.km1 * B.km1.dag(), B.get_states()))[-1] -
            #           np.real(expect(B.k0 * B.k0.dag(), B.get_states()))[-1])

            #ax.plot(B.t_vec, np.sqrt(2) * np.real(expect(B.system.Sx, B.get_states())), '--', label=f'Expect of Sx')
            #ax.plot(B.t_vec, np.sqrt(2) * np.real(expect(B.system.Sy, B.get_states())), label=f'Expect of Sy')
            #ax.plot(B.t_vec, np.real(expect(B.system.Sz ** 2, B.get_states())), label=f'Expect of Sz')
            ax.plot(B.t_vec, np.real(expect(B.k0 * B.k0.dag(), B.get_states())), label=f'Expect of k0')
            ax.plot(B.t_vec, np.real(expect(B.k1 * B.k1.dag(), B.get_states())), label=f'Expect of k1')
            ax.plot(B.t_vec, np.real(expect(B.km1 * B.km1.dag(), B.get_states())), label=f'Expect of km1')
            #ax.plot(B.t_vec, np.sin(BzNoiseOmega * B.t_vec), label=f'Signal')
            ax.plot([], [], label=f'{t_ev}')
            ax.plot([], [], label=f'{B.system.pi_time}')
            plt.grid()
            plt.legend()
            plt.pause(1 / 24)
            plt.cla()
    ax.plot(B.t_vec, np.real(expect(B.k0 * B.k0.dag(), B.get_states())), label=f'Expect of k0')
    ax.plot(B.t_vec, np.real(expect(B.k1 * B.k1.dag(), B.get_states())), label=f'Expect of k1')
    ax.plot(B.t_vec, np.real(expect(B.km1 * B.km1.dag(), B.get_states())), label=f'Expect of km1')
    plt.show()


def perform_pulsed_odmr(n):
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    f_arr = np.linspace(-300e6, 300e6, 200)
    BzSignalAmp = 0
    BzSignalOmega = np.pi / 2
    BzSignalPhase = 0
    BzNoiseAmp = 0
    BzNoiseOmega = 8 * np.pi * 3e6
    BzNoiseOffset = 0

    N = 1
    total_res = []

    for _N in range(N):
        res = []
        BzNoisePhase = np.random.uniform(0, 2 * np.pi)
        noise = np.array([
            lambda t: 0,
            lambda t: 0,
            lambda t: BzSignalAmp + BzNoiseAmp * np.sin(BzNoiseOmega * t + BzNoisePhase) + BzNoiseOffset
        ])
        for idx, f_shift in enumerate(f_arr):
            print(f'Point {idx + 1}/{len(f_arr)}')
            B = three_level_sim(n_steps=n, collapse=True)
            B.system = B.system.add_magnetic_field(
                [noise])  # Has to be done, since dataclass.replace needs to update the class

            print(type(B.system.H))

            #B.system = B.system.update_system(B1=5, omega=2.87e9, detuning=-12e6)
            #res = B.eigenvals(np.linspace(0, B.system.pi_time, B.n_steps))
            #matplotlib.use('TkAgg')
            #plt.title(f'{B.system.pi_time}')
            #plt.plot(res.states.expect[0])
            #plt.plot(res.states.expect[1])
            #plt.show()





            B.pulsed_odmr(driving_strength=3, freq_shift=f_shift)

            res.append(np.real(expect(B.k0 * B.k0.dag(), B.get_states()))[-1])

            ax.plot(B.t_vec, np.sqrt(2) * np.real(expect(B.system.Sx, B.get_states())), '--', label=f'Expect of Sx')
            ax.plot(B.t_vec, np.sqrt(2) * np.real(expect(B.system.Sy, B.get_states())), label=f'Expect of Sy')
            ax.plot(B.t_vec, np.real(expect(B.system.Sz ** 2, B.get_states())), label=f'Expect of Sz')
            ax.plot(B.t_vec, np.real(expect(B.k0 * B.k0.dag(), B.get_states())), label=f'Expect of k0')
            ax.plot(B.t_vec, np.real(expect(B.k1 * B.k1.dag(), B.get_states())), label=f'Expect of k1')
            ax.plot(B.t_vec, np.real(expect(B.km1 * B.km1.dag(), B.get_states())), label=f'Expect of km1')
            ax.plot(B.t_vec, np.sin(BzNoiseOmega * B.t_vec), label=f'Signal')
            ax.plot([], [], label=f'{f_shift}')
            plt.grid()
            plt.legend()
            plt.pause(1 / 24)
            plt.cla()
        total_res.append(res)

    plt.figure()
    for X in total_res:
        plt.plot(f_arr, np.array(X),
                 label=f'Pi_time: {B.system.pi_time * 1e9:.02f} ns')
        # plt.plot(t_arr, 1 - np.array(res2))
    plt.ylim((-1, 1))
    plt.legend()
    plt.show()


@timing
def main(n):
    #perform_pulsed_odmr(n)
    perform_rabi(n)


if __name__ == '__main__':
    n = 200
    main(n)

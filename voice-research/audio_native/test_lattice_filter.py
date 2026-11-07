import unittest
import numpy as np
from scipy.signal import lfilter
from excitation import GeneratedVoicebox, VocalControls, pulse_profile
from lattice_filter import normalized_lattice
from lpc_coordinates import from_reflection
from streaming_voicebox import VoiceboxConfig


class LatticeFilterTests(unittest.TestCase):
    def test_fixed_filter_matches_direct_impulse_and_random_excitation(self):
        rng=np.random.default_rng(213)
        for order in (1,4,12,32):
            reflection=rng.uniform(-.3,.3,order)
            scale=np.prod(np.sqrt(1-reflection*reflection))
            for excitation in (np.r_[1.,np.zeros(4095)],rng.normal(0,.03,4096)):
                expected=lfilter([1.],from_reflection(reflection),excitation)
                actual=normalized_lattice(excitation/scale,reflection,np.zeros(order))
                np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=2e-10)

    def test_block_boundaries_preserve_fixed_filter_output_and_state(self):
        rng=np.random.default_rng(12);reflection=rng.uniform(-.6,.6,12)
        excitation=rng.normal(0,.03,4096)
        whole_state=np.zeros(12);chunk_state=np.zeros(12)
        whole=normalized_lattice(excitation,reflection,whole_state)
        chunks=[normalized_lattice(part,reflection,chunk_state) for part in np.array_split(excitation,37)]
        np.testing.assert_array_equal(np.concatenate(chunks),whole)
        np.testing.assert_array_equal(chunk_state,whole_state)

    def test_time_varying_junctions_obey_energy_balance(self):
        rng=np.random.default_rng(99);state=rng.normal(0,.1,32)
        for i in range(1000):
            reflection=rng.uniform(-.999,.999,32)
            drive=rng.normal(0,.05)
            before=float(state@state)
            outgoing=reflection[-1]*drive+np.sqrt(1-reflection[-1]**2)*state[-1]
            normalized_lattice(np.array([drive]),reflection,state)
            self.assertAlmostEqual(float(state@state),before+drive*drive-outgoing*outgoing,places=12)

    def test_zero_input_energy_cannot_grow_when_reflection_changes(self):
        rng=np.random.default_rng(43);state=rng.normal(0,.1,32)
        initial=float(state@state)
        for i in range(2000):
            reflection=rng.uniform(-.999,.999,32)
            before=float(state@state)
            output=normalized_lattice(np.zeros(1),reflection,state)
            self.assertLessEqual(float(state@state),before+1e-13)
            self.assertLessEqual(abs(output[0]),np.sqrt(initial)+1e-13)

    def test_fixed_filter_voicebox_matches_existing_synthesis(self):
        rng=np.random.default_rng(22)
        coefficients=from_reflection(rng.uniform(-.2,.2,32))
        config=VoiceboxConfig(order=32)
        direct=GeneratedVoicebox(pulse_profile(64),config)
        lattice=GeneratedVoicebox(pulse_profile(64),config,filter_mode='normalized-lattice')
        for i in range(30):
            control=VocalControls(i*240,240,coefficients,100.+i*5,.85,.003+.001*np.sin(i))
            np.testing.assert_allclose(lattice.push(control),direct.push(control),atol=1e-13,rtol=1e-10)

    def test_bad_state_and_reflection_are_rejected_before_mutation(self):
        state=np.ones(2);original=state.copy()
        for reflection in (np.array([.1,1.]),np.array([.1,np.nan])):
            with self.assertRaises(ValueError):normalized_lattice(np.zeros(3),reflection,state)
            np.testing.assert_array_equal(state,original)
        with self.assertRaises(ValueError):normalized_lattice(np.zeros(3),np.zeros(2),np.zeros(3))
        with self.assertRaises(ValueError):GeneratedVoicebox(filter_mode='unrecognized')

    def test_smooth_scattering_matches_samplewise_angle_rotation(self):
        rng=np.random.default_rng(63);old=rng.uniform(-.9,.9,12);new=rng.uniform(-.9,.9,12)
        white=rng.normal(0,.01,240);actual_state=rng.normal(0,.1,12);expected_state=actual_state.copy()
        actual=normalized_lattice(white,new,actual_state,previous_reflection=old)
        expected=[]
        for n,x in enumerate(white):
            angle=np.arcsin(old)+(n+1)/len(white)*(np.arcsin(new)-np.arcsin(old))
            expected.append(normalized_lattice(np.array([x]),np.sin(angle),expected_state)[0])
        np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=1e-11)
        np.testing.assert_allclose(actual_state,expected_state,atol=2e-14,rtol=1e-11)

    def test_smooth_zero_input_does_not_add_state_energy(self):
        rng=np.random.default_rng(83);state=rng.normal(0,.1,32)
        old=rng.uniform(-.999,.999,32)
        for i in range(100):
            target=rng.uniform(-.999,.999,32);before=float(state@state)
            normalized_lattice(np.zeros(240),target,state,previous_reflection=old)
            self.assertLessEqual(float(state@state),before+1e-13)
            old=target
        before=state.copy()
        with self.assertRaises(ValueError):normalized_lattice(np.zeros(240),old,state,previous_reflection=np.zeros(31))
        np.testing.assert_array_equal(state,before)


if __name__=='__main__':unittest.main()

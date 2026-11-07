import unittest
import numpy as np
from periodic_hold_correction import correct_periodic_hold


class PeriodicHoldTests(unittest.TestCase):
    def test_removes_known_grid_preserves_mean_and_context(self):
        rate=800;period=10;times=np.arange(rate*3)/rate
        pattern=np.stack([np.sin(2*np.pi*np.arange(period)/period),.2*np.cos(4*np.pi*np.arange(period)/period)],axis=1)
        values=np.array([3.,-2.])+pattern[np.arange(len(times))%period]
        # Different context outside the target hold must remain untouched.
        values[times<.5]+=np.stack([times[times<.5],times[times<.5]**2],axis=1)
        output,template,gate,info=correct_periodic_hold(values,period,rate,.6,2.4)
        interior=(times>.8)&(times<2.2)
        np.testing.assert_allclose(output[interior],np.tile([3.,-2.],(sum(interior),1)),atol=1e-12)
        np.testing.assert_array_equal(output[gate==0],values[gate==0])
        np.testing.assert_allclose(template,pattern,atol=1e-12)
        self.assertLess(info['afterTemporalRms'],1e-12)

    def test_zero_and_half_strength_are_exact_controls(self):
        values=np.tile(np.arange(10)[:,None],(240,3)).astype(float)
        full,_,_,_=correct_periodic_hold(values,10,800,.5,2.5,1.)
        half,_,_,_=correct_periodic_hold(values,10,800,.5,2.5,.5)
        zero,_,_,_=correct_periodic_hold(values,10,800,.5,2.5,0.)
        np.testing.assert_array_equal(zero,values)
        np.testing.assert_allclose(half,(values+full)/2,atol=1e-12)

    def test_invalid_or_short_support_rejected(self):
        values=np.ones((2400,2))
        for period,rate,start,end,strength in [(1,800,.5,2.5,1),(10,0,.5,2.5,1),
                (10,800,.5,.6,1),(10,800,.5,4.,1),(10,800,.5,2.5,float('nan'))]:
            with self.assertRaises(ValueError):correct_periodic_hold(values,period,rate,start,end,strength)
        values[5,0]=np.nan
        with self.assertRaises(ValueError):correct_periodic_hold(values,10,800,.5,2.5)


if __name__=='__main__':unittest.main()

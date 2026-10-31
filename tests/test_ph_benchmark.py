import unittest
import numpy as np
from scipy.stats import norm
from ph_benchmark.core import (smooth,block_slopes,bocpd_scores,alarms,event_metrics,generate,SCENARIOS)


class BenchmarkTests(unittest.TestCase):
    def test_no_future_leakage(self):
        rng=np.random.default_rng(123)
        y=rng.normal(size=400)
        altered=y.copy();altered[200:]+=100
        for family,size in [('raw',1),('ema',31),('wsg',61)]:
            np.testing.assert_allclose(smooth(y,family,size)[:200],smooth(altered,family,size)[:200])
        a=bocpd_scores(y,.5);b=bocpd_scores(altered,.5)
        np.testing.assert_allclose(a[:200],b[:200])

    def test_block_slope_units_and_availability(self):
        times,s=block_slopes(11-.02*np.arange(100)/60)
        np.testing.assert_allclose(s,-.02,atol=1e-12)
        np.testing.assert_array_equal(times,[29,59,89])

    def test_bocpd_against_enumeration(self):
        # Enumerate every segmentation for a short sequence, independently of recursion.
        z=np.array([-.8,-.9,.2,.3]);h=.05;sigma=.3
        scores=bocpd_scores(z,sigma)
        for n in range(1,5):
            total=recent=0.
            for bits in range(2**(n-1)):
                cuts=[i for i in range(1,n) if bits & (1<<(i-1))]
                probability=h**len(cuts)*(1-h)**(n-1-len(cuts))
                boundaries=[0]+cuts+[n]
                for start,end in zip(boundaries[:-1],boundaries[1:]):
                    m=-.5;v=1.
                    for value in z[start:end]:
                        probability*=norm.pdf(value,m,np.sqrt(v+sigma**2))
                        new_v=1/(1/v+1/sigma**2)
                        m=new_v*(m/v+value/sigma**2);v=new_v
                total+=probability
                if n-boundaries[-2]<=2: recent+=probability
            self.assertAlmostEqual(scores[n-1],recent/total,places=12)

    def test_matching_does_not_reward_early_or_duplicate_alarms(self):
        m=event_metrics([490,510,520,900],(500,500))
        self.assertEqual(m['detected'],1);self.assertEqual(m['false_alarms'],3)
        self.assertEqual(m['delay_from_onset_seconds'],10)
        self.assertTrue(np.isnan(event_metrics([],(500,500))['delay_from_onset_seconds']))
        self.assertEqual(event_metrics([300],None)['false_alarms'],1)
        self.assertEqual(event_metrics([550],(500,600))['delay_after_interval_seconds'],0)

    def test_reproducibility_and_truth(self):
        p=dict(decline_scale_ph_per_minute=.02,start_ph=11.29,residual_lag1_median=.5,
               positive_residual_sd_median=.003,observed_grid_ph=.01)
        for name in SCENARIOS:
            a=generate(p,name,1,1);b=generate(p,name,1,1)
            np.testing.assert_array_equal(a[2],b[2]);self.assertEqual(a[3],b[3])
            np.testing.assert_allclose(a[2]*100,np.round(a[2]*100))
            if name in ('stable','decline','temporary_disturbance'):self.assertIsNone(a[3])
        t,clean,_,target=generate(p,'abrupt_flattening',1,0)
        np.testing.assert_allclose(clean[target[0]:],clean[target[0]])


if __name__=='__main__':unittest.main()

import itertools
import math
import unittest
from ball_temporal_select import Settings, select_path


def c(x,mass=.5,rank=1,y=0):
    return dict(x=x,y=y,peak_mass=mass,rank=rank)


class SelectionTests(unittest.TestCase):
    def test_secondary_candidate_beats_isolated_teleport(self):
        groups={0:[c(0)],1:[c(500,.7),c(8,.3,2)],2:[c(16)]}
        path,_=select_path([0,1,2],groups,60,Settings())
        self.assertEqual([p['x'] for p in path],[0,8,16])

    def test_bounce_and_correct_raw_preserved(self):
        path,_=select_path([0,1,2],{0:[c(0)],1:[c(8)],2:[c(0)]},60,Settings())
        self.assertEqual([p['rank'] for p in path],[1,1,1])

    def test_missing_and_weak_candidates_allow_gaps(self):
        path,_=select_path([0,1,2],{0:[c(0)],1:[c(500,1e-12)],2:[c(16)]},60,Settings())
        self.assertIsNone(path[1])
        self.assertIsNotNone(path[0])
        self.assertIsNotNone(path[2])
        path,_=select_path([0,1],{},60,Settings())
        self.assertEqual(path,[None,None])

    def test_fps_and_frame_spacing(self):
        a,ca=select_path([0,1],{0:[c(0)],1:[c(50)]},30,Settings())
        b,cb=select_path([0,2],{0:[c(0)],2:[c(50)]},60,Settings())
        self.assertEqual(a,b)
        self.assertAlmostEqual(ca,cb)

    def test_invalid_input(self):
        for frames,groups in [([1,1],{}),([0],{0:[c(float('nan'))]})]:
            with self.assertRaises(ValueError):
                select_path(frames,groups,60,Settings())

    def test_matches_brute_force_objective(self):
        groups={0:[c(0),c(150,.2,2)],1:[c(140),c(15,.3,2)],2:[c(25)]}
        settings=Settings()
        def cost(path):
            total=sum(settings.gap_cost if p is None else -.5*math.log(p['peak_mass']) for p in path)
            for a,b in zip(path,path[1:]):
                total += (0 if a is None and b is None else .75 if a is None or b is None else ((a['x']-b['x'])/32)**2)
            return total
        choices=list(itertools.product(*[groups[i]+[None] for i in range(3)]))
        path,total=select_path([0,1,2],groups,60,settings)
        self.assertAlmostEqual(total,min(map(cost,choices)))
        self.assertAlmostEqual(total,cost(path))


if __name__ == '__main__':
    unittest.main()

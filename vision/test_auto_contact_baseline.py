import unittest
import numpy as np
from auto_contact_baseline import pick_contacts, match_times, evaluate


class ContactTests(unittest.TestCase):
    def test_approach_and_departure(self):
        t=np.arange(121)/60
        d=np.minimum(1.,.05+5*np.abs(t-1))
        events=pick_contacts(t,d)
        self.assertEqual(len(events),1)
        self.assertLess(abs(events[0]['t_s']-1),.04)

    def test_held_ball_not_contact(self):
        t=np.arange(121)/60
        self.assertEqual(pick_contacts(t,np.full(len(t),.05)),[])

    def test_gap_not_bridged(self):
        t=np.arange(121)/60;d=np.minimum(1.,.05+5*np.abs(t-1));d[60]=np.nan
        self.assertEqual(pick_contacts(t,d),[])

    def test_two_contacts(self):
        t=np.arange(181)/60;d=np.minimum(.05+5*np.abs(t-1),.05+5*np.abs(t-2))
        self.assertEqual(len(pick_contacts(t,d)),2)

    def test_one_to_one_matching(self):
        pred=[dict(t_s=.1),dict(t_s=.2)]
        truth=[dict(t_s=.18)]
        self.assertEqual(match_times(pred,truth),[(1,0)])

    def test_unknown_still_counts(self):
        p=[dict(t_s=1.,hitter=dict(proposed=None)),dict(t_s=2.,hitter=dict(proposed='A'))]
        l=[dict(t_s=1.,shot=1,evaluation_hitter='A')]
        s=evaluate(p,l)
        self.assertEqual((s['matched'],s['extra'],len(s['unknown_hitter'])),(1,1,1))


if __name__=='__main__':unittest.main()

import unittest
import numpy as np
from motion_contact_experiment import detect


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.t=np.arange(121)/60
        self.near=np.full(len(self.t),.1)
        self.straight=np.c_[self.t,100+300*self.t,np.full(len(self.t),200)]
        self.turn=np.c_[self.t,100+300*np.abs(self.t-1),np.full(len(self.t),200)]

    def test_straight(self):
        self.assertEqual(detect(self.straight,self.near),[])

    def test_turn(self):
        events=detect(self.turn,self.near)
        self.assertEqual(len(events),1)
        self.assertLess(abs(events[0]['t_s']-1),.06)

    def test_teleport(self):
        jump=self.straight.copy();jump[self.t>=1,1]+=300
        self.assertEqual(detect(jump,self.near),[])

    def test_far_from_paddle(self):
        self.assertEqual(detect(self.turn,np.full(len(self.t),2)),[])


if __name__=='__main__':unittest.main()

"""Deterministic corner and detection regressions; no desktop input."""
import unittest
import cv2
import numpy as np
from navigation import Vision, Planner


class TightCornerTests(unittest.TestCase):
    def test_delayed_control_in_tight_zigzag(self):
        # One full frame of actual input delay, checked at every swept pixel.
        xx=np.arange(2200)
        center=140+.72*np.minimum(xx%400,400-xx%400)
        world=(np.abs(np.arange(526)[:,None]-center)>26).astype(np.uint8)
        distance=cv2.distanceTransform(1-world,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        planner=Planner(margin=2,latency=1/120)
        planner.control_dt=1/120
        y=float(center[315]);held=True
        for scroll in range(0,1200,3):
            action,_,_=planner.choose((315,y,9),distance[:,scroll:scroll+800],360,.72,held)
            for offset in range(1,4):
                yy=y+(-1 if held else 1)*offset*.72
                self.assertGreater(distance[round(yy),scroll+315+offset],9)
            y+=(-1 if held else 1)*3*.72
            held=action

    def test_arrow_near_wall_does_not_erase_wall(self):
        frame=np.full((526,800,3),(145,0,190),np.uint8)
        cv2.rectangle(frame,(334,0),(360,526),(10,0,10),-1)
        triangle=np.array([[297,224],[319,200],[330,236]],np.int32)
        cv2.fillPoly(frame,[triangle],(255,255,255))
        cv2.polylines(frame,[triangle],True,(15,5,15),3)
        vision=Vision(fast=True)
        _,player,clearance=vision.read(frame)
        self.assertIsNotNone(player)
        self.assertEqual(clearance[220,340],0)

    def test_arrow_acquisition_across_search_strip(self):
        for x in (90,315,580):
            frame=np.full((526,800,3),(145,0,190),np.uint8)
            triangle=np.array([[x-15,224],[x+7,200],[x+18,236]],np.int32)
            cv2.fillPoly(frame,[triangle],(255,255,255))
            _,player,_=Vision(fast=True).read(frame)
            self.assertIsNotNone(player)
            self.assertLess(abs(player[0]-x),10)

    def test_high_speed_does_not_clip_latency_to_28_pixels(self):
        _,path,_=Planner(latency=.06).choose(
            (315,200,9),np.full((526,800),100,np.float32),700,1,False)
        self.assertEqual(path[0],(357,242))


if __name__=='__main__':
    cv2.setNumThreads(1)
    unittest.main()

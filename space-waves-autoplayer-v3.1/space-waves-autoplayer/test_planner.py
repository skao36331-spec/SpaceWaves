"""Offline regression tests. No desktop input is sent by these tests."""
import unittest
from pathlib import Path
import cv2
import numpy as np
from navigation import Vision,Planner,Hazard,HazardTracker,WIDTH
from menu_control import Button,Confirmation,buttons_from_ocr,same_button
from autoplayer import Motion,accepted_game_titles,title_matches,selection_is_oversized

ROOT=Path(__file__).resolve().parent


class PlannerTests(unittest.TestCase):
    def test_barrier_uses_recovery_instead_of_stopping(self):
        c=np.zeros((526,800),np.float32)
        action,path,status=Planner().choose((315,250,14),c,300,1,False)
        self.assertIsInstance(action,bool)
        self.assertIn('RECOVERY',status)

    def test_closed_loop_alternating_obstacles(self):
        world=np.zeros((526,4000),np.uint8);world[:50]=1;world[480:]=1
        for x in (600,1800,3000):world[:300,x:x+140]=1
        for x in (1200,2400):world[230:,x:x+140]=1
        distance=cv2.distanceTransform(1-world,cv2.DIST_L2,3)
        planner=Planner();y=250.;held=False
        for scroll in range(0,2800,5):
            action,_,_=planner.choose((315,y,14),distance[:,scroll:scroll+800],300,1,held)
            y+=-5 if action else 5;held=action
            self.assertGreater(float(distance[round(y),scroll+320]),14,f'{scroll=}, {y=}')

    def test_fractional_slope_in_narrow_zigzag(self):
        xx=np.arange(3000);center=140+.72*np.minimum(xx%600,600-xx%600)
        yy=np.arange(526)[:,None]
        world=(np.abs(yy-center[None,:])>35).astype(np.uint8)
        distance=cv2.distanceTransform(1-world,cv2.DIST_L2,3)
        planner=Planner(margin=2,latency=.015);y=float(center[315]);held=True
        for scroll in range(0,1800,5):
            action,_,_=planner.choose((315,y,10),distance[:,scroll:scroll+800],300,.72,held)
            y+=(-5 if action else 5)*.72;held=action
            self.assertGreater(float(distance[round(y),scroll+320]),10,f'{scroll=}, {y=}')

    def test_arrow_outline_does_not_block_its_own_route(self):
        im=np.full((526,800,3),(145,0,190),np.uint8)
        triangle=np.array([[297,224],[319,200],[330,236]],np.int32)
        cv2.fillPoly(im,[triangle],(255,255,255))
        cv2.polylines(im,[triangle],True,(15,5,15),3)
        v=Vision();frame,player,c=v.read(im)
        self.assertIsNotNone(player)
        _,_,status=Planner().choose(player,c,300,1,False)
        self.assertEqual(status,'Route found')

    def test_arrow_still_detected_when_touching_a_known_gear(self):
        # In a tight gap, the arrow's white triangle can sit right against a
        # gear's white disk. Without erasing the gear's last-seen footprint
        # first, the fused blob fails the 3-vertex shape test and the arrow
        # is lost right where tracking matters most.
        im=np.full((526,800,3),(145,0,190),np.uint8)
        triangle=np.array([[297,224],[319,200],[330,236]],np.int32)
        cv2.fillPoly(im,[triangle],(255,255,255))
        cv2.polylines(im,[triangle],True,(15,5,15),3)
        cv2.circle(im,(345,220),18,(255,255,255),-1)
        v=Vision()
        v.hazards=[Hazard(345.,220.,18.)]  # as if detected on the prior frame
        _,player,_=v.read(im)
        self.assertIsNotNone(player)

    def test_precise_distance_transform_more_accurate_on_a_diagonal_gap(self):
        # A single blocking pixel; check the measured distance to a point
        # diagonally offset from it against the true Euclidean distance.
        # The 3x3-mask approximation used previously has larger diagonal
        # error than the precise transform now used in geometry().
        static=np.zeros((200,200),np.uint8);static[100,100]=1
        approx=cv2.distanceTransform(1-static,cv2.DIST_L2,3)
        precise=cv2.distanceTransform(1-static,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        py,px=130,130
        truth=np.hypot(py-100,px-100)
        self.assertLess(abs(float(precise[py,px])-truth),abs(float(approx[py,px])-truth))

    def test_fast_mode_is_opt_in_and_still_finds_the_same_route(self):
        # Vision() itself still defaults to precise (the CLI now passes
        # fast=True by default, but the class default stays safe/exact for
        # any other caller); --fast must still produce a usable route.
        self.assertFalse(Vision().fast)
        im=np.full((526,800,3),(145,0,190),np.uint8)
        triangle=np.array([[297,224],[319,200],[330,236]],np.int32)
        cv2.fillPoly(im,[triangle],(255,255,255))
        cv2.polylines(im,[triangle],True,(15,5,15),3)
        v=Vision(fast=True);frame,player,c=v.read(im)
        self.assertIsNotNone(player)
        _,_,status=Planner().choose(player,c,300,1,False)
        self.assertEqual(status,'Route found')

    def test_geometry_reuses_caller_supplied_hsv(self):
        # read() computes hsv once and must not force geometry() to redo the
        # conversion; passing it through must give the identical result.
        im=np.full((300,400,3),(145,0,190),np.uint8)
        v1=Vision();v2=Vision()
        hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        a=v1.geometry(im,None)
        b=v2.geometry(im,None,hsv)
        self.assertTrue(np.array_equal(a,b))


class MotionTests(unittest.TestCase):
    def test_hanning_window_is_cached_across_calls_of_the_same_size(self):
        motion=Motion()
        frame=np.random.randint(0,255,(300,400,3),np.uint8)
        motion.update(frame,None,1.0)
        motion.update(frame,None,1.02)
        cached=motion._window
        self.assertIsNotNone(cached)
        motion.update(frame,None,1.04)
        self.assertIs(motion._window,cached)

    def test_hanning_window_is_recreated_when_frame_size_changes(self):
        motion=Motion()
        small=np.random.randint(0,255,(300,400,3),np.uint8)
        big=np.random.randint(0,255,(600,800,3),np.uint8)
        motion.update(small,None,1.0)
        motion.update(small,None,1.02)
        first=motion._window
        motion.update(big,None,1.04)
        motion.update(big,None,1.06)
        self.assertIsNot(motion._window,first)


class WindowTitleTests(unittest.TestCase):
    def test_default_titles_match_crazygames_style_tabs(self):
        titles=accepted_game_titles()
        self.assertTrue(title_matches('Space Waves - Play on CrazyGames',titles))
        self.assertTrue(title_matches('spacewaves.io',titles))
        self.assertFalse(title_matches('Some Other Game',titles))

    def test_extra_title_is_additive_not_a_replacement(self):
        titles=accepted_game_titles('Play on MSN')
        # the new site's title still matches...
        self.assertTrue(title_matches('Space Waves | Play on MSN',titles))
        # ...and existing CrazyGames-style tabs must still work unchanged.
        self.assertTrue(title_matches('Space Waves - Play on CrazyGames',titles))
        self.assertFalse(title_matches('Unrelated Browser Tab',titles))

    def test_extra_title_matching_is_case_insensitive(self):
        titles=accepted_game_titles('OzGames')
        self.assertTrue(title_matches('Space Waves - OzGames',titles))


class SelectionSizeTests(unittest.TestCase):
    def test_tight_selection_is_not_flagged(self):
        self.assertFalse(selection_is_oversized(900,650,1920,1080))

    def test_oversized_selection_is_flagged(self):
        self.assertTrue(selection_is_oversized(1900,1060,1920,1080))


class PlannerTests2(unittest.TestCase):
    def test_moving_gear_prediction_blocks_future_position(self):
        planner=Planner();planner.hazards=[Hazard(500,200,20,0,100,10)]
        c=np.full((526,800),100,np.float32)
        future=planner._field(c,300,400)
        self.assertLess(future[250,500],0)
        self.assertGreater(future[150,500],30)

    def test_decelerating_gear_grows_cautious_past_predicted_reversal(self):
        # Same gear, but decelerating hard toward a reversal (large negative ay
        # while moving down). Past the predicted turning point the straight-line
        # extrapolation is unreliable, so clearance there should be lower than
        # the constant-velocity case above at the same lookahead.
        c=np.full((526,800),100,np.float32)
        coasting=Planner();coasting.hazards=[Hazard(500,200,20,0,100,age=10)]
        reversing=Planner();reversing.hazards=[Hazard(500,200,20,0,100,age=10,ay=-300)]
        far_coast=coasting._field(c,300,400)
        far_reverse=reversing._field(c,300,400)
        # Point directly under the predicted (still-linear) gear position, at a
        # lookahead well past the estimated turning point (~0.33s here).
        y,x=250,500
        self.assertLess(far_reverse[y,x],far_coast[y,x])

    def test_tracker_subtracts_scroll(self):
        t=HazardTracker();t.update([Hazard(500,200,20)],1,300)
        g=t.update([Hazard(490,204,20)],1+1/30,300)[0]
        self.assertAlmostEqual(g.vx,0,places=4)
        self.assertGreater(g.vy,0)


class MenuTests(unittest.TestCase):
    def rows(self,text,score=.99):return [[[[200,200],[400,200],[400,240],[200,240]],text,score]]
    def test_allowed_labels(self):
        for label in ('NEXT','Next Level','GIVE UP','Giveup','Retry'):
            self.assertEqual(len(buttons_from_ocr(self.rows(label),(526,800,3))),1)
    def test_unrelated_and_uncertain_text_ignored(self):
        for label in ('Buy','Next offer','Watch ad','Level 3','Give'):
            self.assertFalse(buttons_from_ocr(self.rows(label),(526,800,3)))
        self.assertFalse(buttons_from_ocr(self.rows('Next',.60),(526,800,3)))
    def test_split_two_word_label_is_recombined(self):
        # A tight/small button can get OCR'd as two separate regions for one
        # label ('GIVE' then 'UP'); neither half alone is an allowed label,
        # so they must be merged before matching.
        rows=[[[[200,200],[260,200],[260,240],[200,240]],'GIVE',.97],
              [[[268,200],[310,200],[310,240],[268,240]],'UP',.97]]
        buttons=buttons_from_ocr(rows,(526,800,3))
        self.assertEqual(len(buttons),1)
        self.assertEqual(buttons[0].label,'giveup')
    def test_split_label_does_not_merge_across_unrelated_far_text(self):
        # A same-row-ish score counter far to the side must not glue onto an
        # unrelated word and produce a false label.
        rows=[[[[10,200],[60,200],[60,240],[10,240]],'3446',.99],
              [[[400,200],[480,200],[480,240],[400,240]],'Give',.97]]
        self.assertFalse(buttons_from_ocr(rows,(526,800,3)))
    def test_confirmation_and_cooldown(self):
        c=Confirmation();b=Button('next',(200,200,100,30),.99)
        self.assertFalse(c.observe(b,1));self.assertTrue(c.observe(b,1.6));c.clicked(1.6)
        self.assertFalse(c.observe(b,2.2));self.assertFalse(c.observe(b,2.8))
        self.assertTrue(c.observe(b,3.7))
    def test_stale_button_pixels_rejected(self):
        b=Button('next',(200,200,100,30),.99)
        a=np.zeros((526,800,3),np.uint8);new=a.copy();new[194:236,194:306]=255
        self.assertTrue(same_button(b,a,a));self.assertFalse(same_button(b,a,new))
    def test_one_detection_never_clicks(self):
        c=Confirmation();b=Button('giveup',(200,200,100,30),.99)
        self.assertFalse(c.observe(b,1));self.assertFalse(c.observe(None,1.6))
        self.assertFalse(c.observe(b,2.2))


def _prepare(path):
    image=cv2.imread(str(path))
    h=round(image.shape[0]*WIDTH/image.shape[1])
    return cv2.resize(image[:,:,:3],(WIDTH,h),interpolation=cv2.INTER_AREA)


class RealMenuScreenshotTests(unittest.TestCase):
    """End-to-end check against the actual 'Give up' and 'Next' screens
    supplied by the user, run through the real OCR engine (not synthetic
    rows). Confirms the bot recognizes and would click the right button,
    ignoring the Revive/Restart buttons that also appear on-screen."""

    @classmethod
    def setUpClass(cls):
        giveup_path=ROOT/'sample-giveup-screen.png'
        next_path=ROOT/'sample-next-screen.png'
        if not (giveup_path.exists() and next_path.exists()):
            raise unittest.SkipTest('Sample screenshots not present')
        try:
            from rapidocr_onnxruntime import RapidOCR
        except Exception:
            raise unittest.SkipTest('rapidocr_onnxruntime not installed')
        cls.engine=RapidOCR(intra_op_num_threads=1,inter_op_num_threads=1,use_cls=False)
        cls.giveup_frame=_prepare(giveup_path)
        cls.next_frame=_prepare(next_path)

    def _detect(self,frame):
        rows,_=self.engine(frame,use_cls=False)
        return buttons_from_ocr(rows,frame.shape)

    def test_give_up_screen_detected_and_prioritized_over_revive(self):
        buttons=self._detect(self.giveup_frame)
        self.assertTrue(buttons,'No menu button detected on the Give up screen')
        self.assertEqual(buttons[0].label,'giveup')
        # Revive is an icon+video-ad button; it must never be the click target.
        self.assertNotIn('revive',[b.label for b in buttons])

    def test_next_screen_detected_and_prioritized_over_restart(self):
        buttons=self._detect(self.next_frame)
        self.assertTrue(buttons,'No menu button detected on the Level Complete screen')
        self.assertEqual(buttons[0].label,'next')

    def test_give_up_confirms_and_survives_pixel_check_across_frames(self):
        buttons=self._detect(self.giveup_frame)
        button=buttons[0]
        confirm=Confirmation()
        self.assertFalse(confirm.observe(button,10.0))
        self.assertTrue(confirm.observe(button,10.5))
        self.assertTrue(same_button(button,self.giveup_frame,self.giveup_frame))

    def test_next_confirms_and_survives_pixel_check_across_frames(self):
        buttons=self._detect(self.next_frame)
        button=buttons[0]
        confirm=Confirmation()
        self.assertFalse(confirm.observe(button,10.0))
        self.assertTrue(confirm.observe(button,10.5))
        self.assertTrue(same_button(button,self.next_frame,self.next_frame))

    def test_next_screen_detected_across_a_different_color_theme(self):
        # Space Waves reskins its Level Complete screen per theme (green vs
        # purple here); the label text and layout are what matter, not the palette.
        path=ROOT/'sample-next-purple-screen.png'
        if not path.exists():
            raise unittest.SkipTest('Sample screenshot not present')
        frame=_prepare(path)
        buttons=self._detect(frame)
        self.assertTrue(buttons,'No menu button detected on the purple Level Complete screen')
        self.assertEqual(buttons[0].label,'next')


class RealFollowPromptScreenshotTest(unittest.TestCase):
    """The CrazyGames 'Follow on TikTok' interstitial shows a 'Continue playing'
    button alongside a 'Follow CrazyGames' button; only the former may be
    clicked."""

    @classmethod
    def setUpClass(cls):
        path=ROOT/'sample-continueplaying-screen.png'
        if not path.exists():
            raise unittest.SkipTest('Sample screenshot not present')
        try:
            from rapidocr_onnxruntime import RapidOCR
        except Exception:
            raise unittest.SkipTest('rapidocr_onnxruntime not installed')
        cls.engine=RapidOCR(intra_op_num_threads=1,inter_op_num_threads=1,use_cls=False)
        cls.frame=_prepare(path)

    def test_continue_playing_detected_and_follow_button_ignored(self):
        rows,_=self.engine(self.frame,use_cls=False)
        buttons=buttons_from_ocr(rows,self.frame.shape)
        self.assertTrue(buttons,'Continue playing button not detected')
        self.assertEqual(buttons[0].label,'continueplaying')
        self.assertNotIn('followcrazygames',[b.label for b in buttons])

    def test_continue_playing_confirms_across_frames(self):
        rows,_=self.engine(self.frame,use_cls=False)
        button=buttons_from_ocr(rows,self.frame.shape)[0]
        confirm=Confirmation()
        self.assertFalse(confirm.observe(button,10.0))
        self.assertTrue(confirm.observe(button,10.5))
        self.assertTrue(same_button(button,self.frame,self.frame))


class RealClaimAdScreenshotTest(unittest.TestCase):
    """The 'Level Complete!' claim-ad interstitial shows a 'Claim 2X' video-ad
    button alongside 'No thanks'; only the latter may be clicked."""

    @classmethod
    def setUpClass(cls):
        path=ROOT/'sample-nothanks-screen.png'
        if not path.exists():
            raise unittest.SkipTest('Sample screenshot not present')
        try:
            from rapidocr_onnxruntime import RapidOCR
        except Exception:
            raise unittest.SkipTest('rapidocr_onnxruntime not installed')
        cls.engine=RapidOCR(intra_op_num_threads=1,inter_op_num_threads=1,use_cls=False)
        cls.frame=_prepare(path)

    def test_no_thanks_detected_and_claim_ad_ignored(self):
        rows,_=self.engine(self.frame,use_cls=False)
        buttons=buttons_from_ocr(rows,self.frame.shape)
        self.assertTrue(buttons,'No thanks button not detected')
        self.assertEqual(buttons[0].label,'nothanks')
        self.assertNotIn('claimx',[b.label for b in buttons])

    def test_no_thanks_confirms_across_frames(self):
        rows,_=self.engine(self.frame,use_cls=False)
        button=buttons_from_ocr(rows,self.frame.shape)[0]
        confirm=Confirmation()
        self.assertFalse(confirm.observe(button,10.0))
        self.assertTrue(confirm.observe(button,10.5))
        self.assertTrue(same_button(button,self.frame,self.frame))


class MenuWorkerResilienceTest(unittest.TestCase):
    """A single bad frame must not permanently disable menu detection for
    the rest of the session, but should still surface as a transient error
    and clear once a good frame follows."""

    def test_one_bad_frame_does_not_stop_the_worker(self):
        from menu_control import MenuWorker
        worker=MenuWorker.__new__(MenuWorker)  # skip starting the real thread
        worker.results=__import__('queue').Queue(maxsize=1)
        worker.error=None
        calls={'n':0}
        def flaky_engine(frame,use_cls=False):
            calls['n']+=1
            if calls['n']==1:raise RuntimeError('simulated transient OCR failure')
            return [],None
        with self.assertRaises(RuntimeError):
            worker._process_one(flaky_engine,np.zeros((10,10,3),np.uint8),1.0)
        worker.error='simulated transient OCR failure'
        # The second frame must still be processable -- nothing about the
        # first failure should have left the worker or its queues broken.
        worker._process_one(flaky_engine,np.zeros((10,10,3),np.uint8),2.0)
        button,frame,stamp=worker.results.get_nowait()
        self.assertIsNone(button);self.assertEqual(stamp,2.0)


if __name__=='__main__':
    cv2.setNumThreads(1);unittest.main()

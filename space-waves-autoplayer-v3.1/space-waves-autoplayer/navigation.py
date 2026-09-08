"""Visual geometry, hazard tracking, and fractional trajectory planning for v2."""
from dataclasses import dataclass
import time
import cv2
import numpy as np

WIDTH=800

@dataclass
class Hazard:
    x:float
    y:float
    radius:float
    vx:float=0.0  # residual motion after subtracting course scrolling
    vy:float=0.0
    age:int=1
    ax:float=0.0  # smoothed rate of change of vx/vy, used to anticipate reversals
    ay:float=0.0


class Vision:
    def __init__(self, fast=False):
        self.last=None
        self.misses=0
        self.hazards=[]
        self.blocked=None
        self.background=180.0
        # Precise diagonal clearance costs real CPU on weak machines; --fast
        # trades that back for the cheaper, still-reasonable approximation.
        self.fast=fast

    def geometry(self, frame, player_component=None, hsv=None):
        if hsv is None:hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        white=((hsv[:,:,1]<55)&(hsv[:,:,2]>215)).astype(np.uint8)
        saturated=(hsv[:,:,1]>100).astype(np.uint8)
        hist=cv2.calcHist([hsv],[2],saturated,[256],[0,256]).ravel()
        if hist.sum()>frame.shape[0]*frame.shape[1]*.15:
            self.background=float(np.searchsorted(hist.cumsum(),hist.sum()*.74))
        blocked=((hsv[:,:,2]<self.background*.74)|(white>0)).astype(np.uint8)
        blocked=cv2.morphologyEx(blocked,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        # A complete enclosing disk blocks every rotation of a standalone gear.
        contours,_=cv2.findContours(white,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        hazards=[]
        for contour in contours:
            area=cv2.contourArea(contour)
            if not 110<area<15000:continue
            bx,by,bw,bh=cv2.boundingRect(contour)
            if not .70<bw/bh<1.43:continue
            perimeter=cv2.arcLength(contour,True)
            approx=cv2.approxPolyDP(contour,.022*perimeter,True)
            if len(approx)<8:continue
            (cx,cy),radius=cv2.minEnclosingCircle(contour)
            if not 9<radius<85:continue
            if not radius+3<cx<frame.shape[1]-radius-3:continue
            if not radius+3<cy<frame.shape[0]-radius-3:continue
            if player_component is not None:
                px,py,pw,ph=player_component
                if px-3<cx<px+pw+3 and py-3<cy<py+ph+3:continue
            # Only separate gears qualify for motion tracking. Nearby walls must
            # not be erased when removing the gear's current position.
            angles=np.linspace(0,2*np.pi,40,endpoint=False)
            rx=np.rint(cx+(radius+4)*np.cos(angles)).astype(int)
            ry=np.rint(cy+(radius+4)*np.sin(angles)).astype(int)
            if np.mean(blocked[ry,rx])>.12:continue
            hazards.append(Hazard(cx,cy,radius+1))
        static=blocked.copy()
        for g in hazards:
            cv2.circle(static,(round(g.x),round(g.y)),int(np.ceil(g.radius+1)),0,-1)
            cv2.circle(blocked,(round(g.x),round(g.y)),int(np.ceil(g.radius)),1,-1)
        if player_component is not None:
            bx,by,bw,bh=player_component
            # Remove the triangle and its outline, not a full rectangle of nearby
            # course geometry. Never clear a large box through a wall or spike.
            region=white[by:by+bh,bx:bx+bw]
            cs,_=cv2.findContours(region,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if cs:
                c=max(cs,key=cv2.contourArea)+np.array([[[bx,by]]])
                remove=np.zeros_like(blocked)
                # The arrow's dark border has sharp tips extending beyond its
                # white triangle. Remove its enclosing contour when isolated.
                # Only inspect the neighborhood which can contain an outline
                # of the allowed size. Avoid scanning the entire course twice.
                left=max(0,bx-25);top=max(0,by-25)
                right=min(frame.shape[1],bx+bw+25);bottom=min(frame.shape[0],by+bh+25)
                dark=(hsv[top:bottom,left:right,2]<self.background*.45).astype(np.uint8)
                outlines,_=cv2.findContours(dark,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                outer=None
                for outline in outlines:
                    outline=outline+np.array([[[left,top]]])
                    ox,oy,ow,oh=cv2.boundingRect(outline)
                    if ox<=left or oy<=top or ox+ow>=right or oy+oh>=bottom:continue
                    if ow>bw+24 or oh>bh+24:continue
                    if cv2.pointPolygonTest(outline,(float(bx+bw/2),float(by+bh/2)),False)>=0:
                        outer=outline;break
                cv2.drawContours(remove,[outer if outer is not None else c],-1,1,-1)
                pad=3 if outer is not None else 9
                remove=cv2.dilate(remove,np.ones((pad,pad),np.uint8))
                blocked[remove>0]=0;static[remove>0]=0
        # Only true outer edges are excluded. v1 discarded the top 5.5%.
        for mask in (blocked,static):
            mask[:2]=1;mask[-2:]=1;mask[:,-2:]=1
        self.blocked=blocked
        self.hazards=hazards
        # Exact Euclidean distance transform instead of the 3x3-mask
        # approximation: the approximation can be off by several percent on
        # diagonal paths, which matters most exactly where it's tightest —
        # narrow diagonal gaps between obstacles. --fast trades this back
        # for the cheaper approximation on weaker machines.
        if self.fast:
            # 5x5 has much smaller diagonal error than 3x3, without the
            # cost of the exact transform across the whole playfield.
            return cv2.distanceTransform(1-static,cv2.DIST_L2,5)
        return cv2.distanceTransform(1-static,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)

    def read(self,image):
        h=round(image.shape[0]*WIDTH/image.shape[1])
        frame=cv2.resize(image[:,:,:3],(WIDTH,h),interpolation=cv2.INTER_AREA)
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        white=((hsv[:,:,1]<45)&(hsv[:,:,2]>220)).astype(np.uint8)
        # A gear's white disk can visually fuse with the arrow's white triangle
        # when they're close together in a tight gap, breaking the 3-vertex
        # shape test below right when detection matters most. Erase last
        # frame's known gear footprints first; geometry() re-detects them for
        # the current frame regardless.
        for g in self.hazards:
            cv2.circle(white,(round(g.x),round(g.y)),int(np.ceil(g.radius+3)),0,-1)
        # Candidates cannot occur outside this horizontal strip. Include
        # half a maximum arrow width so components at its edge stay intact.
        left=max(0,int(.10*WIDTH)-53);right=min(WIDTH,int(.74*WIDTH)+53)
        n,labels,stats,centers=cv2.connectedComponentsWithStats(white[:,left:right])
        candidates=[]
        for i in range(1,n):
            bx,by,bw,bh,area=stats[i];cx,cy=centers[i]
            bx+=left;cx+=left
            if not (.10*WIDTH<cx<.74*WIDTH and .025*h<cy<.975*h):continue
            if not (9<=bw<=52 and 9<=bh<=52 and 80<=area<=1200):continue
            if not (.28<area/(bw*bh)<.88 and .42<bw/bh<2.3):continue
            patch=(labels[by:by+bh,bx-left:bx-left+bw]==i).astype(np.uint8)
            cs,_=cv2.findContours(patch,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            contour=max(cs,key=cv2.contourArea)
            approx=cv2.approxPolyDP(contour,.055*cv2.arcLength(contour,True),True)
            if len(approx)!=3:continue
            hull_area=cv2.contourArea(cv2.convexHull(contour))
            if hull_area and cv2.contourArea(contour)/hull_area<.90:continue
            score=.5*abs(cx-WIDTH*.394)+.015*abs(area-350)
            if self.last:
                score+=.45*abs(cy-self.last[1])+abs(cx-self.last[0])
            candidates.append((score,cx,cy,(bx,by,bw,bh)))
        if not candidates:
            self.misses+=1
            if self.misses>6:self.last=None
            self.hazards=[]
            return frame,None,None
        _,x,y,box=min(candidates)
        self.last=(float(x),float(y));self.misses=0
        clearance=self.geometry(frame,box,hsv)
        # Radius remains tied to the user's observed white-arrow size.
        radius=max(box[2],box[3])*.43+2
        return frame,(float(x),float(y),float(radius)),clearance


class HazardTracker:
    def __init__(self):
        self.previous=[];self.when=None

    def update(self,detected,now,scroll_speed):
        dt=now-self.when if self.when is not None else 0
        used=set()
        if .005<dt<.20:
            for g in detected:
                options=[]
                for i,old in enumerate(self.previous):
                    if i in used or abs(g.radius-old.radius)>max(4,g.radius*.25):continue
                    px=old.x+(old.vx-scroll_speed)*dt
                    py=old.y+old.vy*dt
                    error=np.hypot(g.x-px,g.y-py)
                    if error<max(14,600*dt):options.append((error,i,old))
                if options:
                    _,i,old=min(options,key=lambda o:o[0]);used.add(i)
                    vx=(g.x-old.x)/dt+scroll_speed;vy=(g.y-old.y)/dt
                    if abs(vx)<700 and abs(vy)<700:
                        new_vx=.65*old.vx+.35*vx;new_vy=.65*old.vy+.35*vy
                        # Oscillating gears decelerate before reversing; track that
                        # trend so extrapolation can grow cautious near the turn
                        # instead of assuming the current velocity continues.
                        rax=(new_vx-old.vx)/dt;ray=(new_vy-old.vy)/dt
                        g.vx=new_vx;g.vy=new_vy;g.age=old.age+1
                        if abs(rax)<4000:g.ax=.5*old.ax+.5*rax
                        if abs(ray)<4000:g.ay=.5*old.ay+.5*ray
        self.previous=detected;self.when=now
        return detected


class Planner:
    def __init__(self,margin=2.0,latency=.025):
        self.margin=margin;self.latency=latency
        self.hazards=[]
        self.control_dt=1/60

    def _field(self,clearance,x,vx):
        # Future course coordinates: scroll itself is represented by increasing x.
        field=clearance
        if self.hazards:
            field=clearance.copy()
            ys=np.arange(field.shape[0],dtype=np.float32)[:,None]
            xs=np.arange(field.shape[1],dtype=np.float32)[None,:]
            t=np.clip((xs-x)/max(60,vx),0,1.5)
            for g in self.hazards:
                # Do not extrapolate a newly discovered gear from one sample.
                ux=g.vx if g.age>=3 else 0
                uy=g.vy if g.age>=3 else 0
                # Limit linear extrapolation when a moving gear may reverse.
                dt=np.minimum(t,.55)
                gx=g.x+ux*dt;gy=g.y+uy*dt
                uncertainty=np.minimum(5,t*3) if g.age>=3 else np.minimum(3,t*2)
                # A decelerating gear (age-qualified acceleration estimate) is
                # heading toward a direction reversal; positions predicted past
                # that turning point trust the straight-line estimate less, so
                # uncertainty grows there instead of assuming it keeps coasting.
                if g.age>=4:
                    ax,ay=g.ax,g.ay
                    turn=min([abs(u/a) for u,a in ((ux,ax),(uy,ay)) if abs(a)>1.0]+[np.inf])
                    if turn<.55:
                        uncertainty=uncertainty+np.clip((dt-turn)*40,0,20)
                d=np.sqrt((xs-gx)**2+(ys-gy)**2)-g.radius-uncertainty
                np.minimum(field,d,out=field)
        return field

    def choose(self,player,clearance,vx,slope,held):
        x,y,r=player;h,w=clearance.shape
        if not (0<=x<w and 0<=y<h):return held,[],"RECOVERY: invalid position"
        field=self._field(clearance,x,vx)
        lead=min(w-8-x,max(0,vx*self.latency))
        startx=min(w-8,x+lead)
        starty=float(np.clip(y+(-1 if held else 1)*lead*slope,1,h-2))
        # Action spacing cannot be faster than approximately one capture cycle.
        step=max(2,min(w//4,int(np.ceil(vx*self.control_dt))))
        visible_right=min(w-2,int(w*.925))  # fixed right-hand decorative rail
        xs=np.arange(round(startx),visible_right-int(np.ceil(r+self.margin))-step-2,step,dtype=int)
        if len(xs)<3:return held,[],"RECOVERY: course edge"
        ys=np.arange(h,dtype=np.float32)
        delta=float(step*np.clip(slope,.25,3))
        transitions=[]
        for sign in (-1,1):
            dest=ys+sign*delta
            lo=np.clip(np.floor(dest).astype(int),0,h-1)
            hi=np.clip(lo+1,0,h-1)
            frac=dest-np.floor(dest)
            transitions.append((dest,lo,hi,frac))
        radius=r+self.margin
        current=max(0,float(field[round(y),round(x)]))
        # Margin only relaxes when already touching geometry, then recovers fast.
        start_margin=min(radius,max(r*.40,current-1))
        cost=np.zeros((2,h),np.float32)
        choices=np.zeros((len(xs),2,h),np.uint8)
        first_cost=None
        BIG=1e7
        # Build the swept-clearance table in bulk. Only the backward recurrence
        # remains sequential; this keeps fine planning affordable at high FPS.
        required=np.minimum(radius,start_margin+(xs-x)*.6)[:,None]
        tables=[]
        columns=xs[:,None]
        endcols=np.minimum(w-1,columns+step)
        for dest,lo,hi,frac in transitions:
            dist=np.minimum(field[lo[None,:],endcols],field[hi[None,:],endcols])
            dist=np.minimum(dist,field[ys.astype(int)[None,:],columns])
            for portion in np.linspace(0,1,max(2,int(np.ceil(step/2)))+1)[1:-1]:
                mid=np.clip(np.rint(ys+(dest-ys)*portion).astype(int),0,h-1)
                dist=np.minimum(dist,field[mid[None,:],np.minimum(w-1,columns+round(step*portion))])
            bad=(dist<required)|((dest<1)|(dest>=h-1))[None,:]
            local=2.2/(1+np.maximum(0,dist-radius))
            fail=BIG+(len(xs)-np.arange(len(xs)))[:,None]*1000+np.maximum(0,required-dist)*10
            tables.append((bad,local,fail))
        for j in range(len(xs)-1,-1,-1):
            opts=[]
            for action,(_,lo,hi,frac) in enumerate(transitions):
                future=cost[action,lo]*(1-frac)+cost[action,hi]*frac
                bad,local,fail=tables[action]
                opts.append(np.where(bad[j],fail[j],future+local[j]))
            new=np.empty_like(cost)
            for previous in (0,1):
                a=opts[0]+(.12 if previous else 0)
                b=opts[1]+(.12 if not previous else 0)
                choices[j,previous]=b<a
                new[previous]=np.minimum(a,b)
            cost=new
            if j==0:first_cost=opts
        previous=0 if held else 1
        # Interpolate action costs at the actual subpixel position, not the
        # rounded policy bit. Rounding the bit can flip a tight turn early.
        lo=int(np.floor(starty));hi=min(h-1,lo+1);fraction=starty-lo
        scores=[float(v[lo]*(1-fraction)+v[hi]*fraction)+
                (.12 if action!=previous else 0)
                for action,v in enumerate(first_cost)]
        first=int(scores[1]<scores[0])
        risky=scores[first]>=BIG
        path=[(int(startx),int(round(starty)))];pos=starty
        for j,xx in enumerate(xs):
            action=first if j==0 else int(choices[j,previous,int(np.clip(round(pos),0,h-1))])
            pos+=(-1 if action==0 else 1)*delta
            if not 1<=pos<h-1:break
            path.append((int(xx+step),round(pos)));previous=action
        return first==0,path,("RECOVERY: best survival route" if risky else "Route found")

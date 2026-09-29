# Pixel-level refinement at FULL resolution around the coarse handwriting: own colour (3x3) +
# 4-direction arbitration + overlap detection.
import cv2, numpy as np
BAND=3; LENGTHS=(2,4,6); DARK=1.35; BRIDGE=True
def dir_kernels(L, band=1):
    """4 directional band kernels (horizontal, vertical, 2 diagonals), `band` px wide, L long,
    centre (band x band) block excluded so a pixel is judged by its neighbours along the line."""
    ks=[]
    base=np.zeros((L,L),np.float32); c=L//2
    h=base.copy(); h[c-band//2:c+band//2+1,:]=1
    v=h.T.copy()
    d=np.zeros((L,L),np.float32)
    for i in range(L):
        for o in range(-(band//2),band//2+1):
            j=i+o
            if 0<=j<L: d[i,j]=1
    a=np.fliplr(d).copy()
    for kk in (h,v,d,a):
        kk=kk.copy(); kk[c-band//2:c+band//2+1,c-band//2:c+band//2+1]=0; ks.append(kk)
    return ks
def od(img):
    f=img.astype(np.float32); H,W=img.shape[:2]
    sm=cv2.resize(f,None,fx=.125,fy=.125,interpolation=cv2.INTER_AREA)
    pm=cv2.dilate(cv2.medianBlur(sm.astype(np.uint8),15),np.ones((9,9),np.uint8)).astype(np.float32)
    P=cv2.GaussianBlur(cv2.resize(pm,(W,H)),(0,0),8)
    OD=-np.log(np.clip((f+1)/(P+1),1e-3,1)); return OD, OD.mean(-1)
def refine(img, coarseHw, printRefMap, coarsePrint=None, fragments=None):
    coarse=coarseHw
    H,W=img.shape[:2]; k=max(5,int(max(H,W)/450))|1
    OD,odm=od(img); ink=odm>0.30
    zone=cv2.dilate(coarseHw.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4*k+1,4*k+1)))>0
    w=(ink&(odm<1.6)).astype(np.float32)
    sb=cv2.boxFilter(OD[...,0]*w,-1,(3,3)); sr=cv2.boxFilter(OD[...,2]*w,-1,(3,3)); sw=cv2.boxFilter(w,-1,(3,3))
    r=np.where(sw>0.3, sb/(sr+1e-3), np.nan)
    core=cv2.erode(coarseHw.astype(np.uint8),np.ones((3,3),np.uint8))>0
    pv=r[core&ink&~np.isnan(r)]; penRef=float(np.median(pv)) if pv.size>50 else float(np.nanmedian(printRefMap))-0.1
    thr=(penRef+printRefMap)/2
    decided=~np.isnan(r)
    ownPen=ink&decided&(np.nan_to_num(r,nan=9)<thr)
    ownPrint=ink&decided&(np.nan_to_num(r,nan=0)>=thr)
    # confidently print-coloured: as close to the local print colour as print itself is
    printStrong=ink&decided&(np.nan_to_num(r,nan=0)>=printRefMap-0.02)
    # too dark to have a colour (sensor-clipped) but outside the coarse handwriting = print: the
    # coarse pass decided those strokes as a whole (black toner clips — 27% of ink on a real scan)
    # clipped cores are judged as whole blobs: a blob reaching well outside the coarse
    # handwriting is a print stroke's body (the pen merely touches/crosses it); pen cores lie
    # inside the coarse handwriting
    clipped=(ink&~decided).astype(np.uint8)
    nC,lC,stC,_=cv2.connectedComponentsWithStats(clipped,8)
    outside=np.bincount(lC.ravel(),weights=(clipped.astype(bool)&~coarse).ravel(),minlength=nC)
    blobPrint=outside>=0.3*np.maximum(stC[:,4],1); blobPrint[0]=False
    printStrong|=blobPrint[lC]
    # 4-direction arbitration
    inkF=ink.astype(np.float32); opF=ownPen.astype(np.float32); orF=printStrong.astype(np.float32)   # 'print runs through' = CONFIDENT print only
    printThrough=np.zeros(ink.shape,bool); penThrough=np.zeros(ink.shape,bool); penVotes=np.zeros(ink.shape,np.int32)
    for m in LENGTHS:                      # several lengths: short and long runs both count
        L=(m*k)|1
        for kk in dir_kernels(L,BAND):
            full=kk.sum()
            n=cv2.filter2D(inkF,-1,kk,borderType=cv2.BORDER_CONSTANT)
            pen=cv2.filter2D(opF,-1,kk,borderType=cv2.BORDER_CONSTANT)/np.maximum(n,1)
            pr=cv2.filter2D(orF,-1,kk,borderType=cv2.BORDER_CONSTANT)/np.maximum(n,1)
            runs=n>=0.6*full/BAND           # the line is actually inked along its length
            printThrough|=(pr>=0.6)&runs
            penThrough|=(pen>=0.6)&runs
            if m==LENGTHS[0]: penVotes+=(pen>=0.5)
    penMost=penVotes>=3
    # the working-size (coarse) result is the base — per-pixel colour is unreliable on small
    # scans (measured: a 1312 px scan lost most pen pixels when colour alone decided). Pixels only
    # leave it where a print stroke runs through them and they are print-coloured (print, or
    # overlap if darker — below); pen-coloured pixels running along pen strokes are added.
    coarseInk=coarse&ink
    # leaves the handwriting when a confident print stroke runs through it and pen does not
    # dominate — including sensor-clipped pixels (no colour of their own), which only the
    # directions can judge
    carve=printThrough&~penMost&(printStrong|(~decided&~penThrough))
    # small detached pieces of the coarse handwriting (lead-in curls, dots, dashes, accents) were
    # put there by their context — never carve them back out on their own colour
    nF,lF,stF,_=cv2.connectedComponentsWithStats(coarse.astype(np.uint8),8)
    smallPiece=stF[:,4]<=(4*k)**2; smallPiece[0]=False
    carve&=~smallPiece[lF]
    if fragments is not None: carve&=~fragments   # placed by context in the coarse pass
    added=((ownPen|~decided)&penThrough) | (ownPrint&penMost&~printThrough)
    hw=zone&ink&(coarseInk|added)&~carve     # carved pixels must not be re-added
    # overlap = pen over print: a print stroke runs straight through AND the pixel is much darker
    # than the pen around it (ink densities add where two inks stack)
    pw=ownPen.astype(np.float32); win=4*k+1
    penLocal=cv2.boxFilter(odm*pw,-1,(win,win))/np.maximum(cv2.boxFilter(pw,-1,(win,win)),1e-6)
    # print must CONTINUE on both sides beyond the pen (a hand-drawn underline has pen, not
    # print, at both ends), within about two stroke widths, in any of the 4 directions
    pvF=(printStrong&~hw).astype(np.float32); both=np.zeros(hw.shape,bool); R2=2*k
    for dd in range(4):
        sa=cv2.filter2D(pvF,-1,half_kernels(R2,BAND,dd,-1),borderType=cv2.BORDER_CONSTANT)
        sb=cv2.filter2D(pvF,-1,half_kernels(R2,BAND,dd,+1),borderType=cv2.BORDER_CONSTANT)
        both|=(sa>=1.5)&(sb>=1.5)
    overlap=hw&both&(odm>DARK*penLocal)
    # only CONFIDENT print is protected: pen pixels merely left out of `hw` must stay erasable
    # (protecting "all other ink" turned every missed pen pixel into permanent print — ghosts)
    base=coarsePrint if coarsePrint is not None else np.zeros_like(hw)
    printLayer=((base|printStrong)&~hw)|overlap
    return hw, overlap, dict(penRef=penRef, printLayer=printLayer, odm=odm)

def half_kernels(L, band, direction, side):
    """Band kernel covering only ONE side (side=-1/+1) of the centre along a direction."""
    full=dir_kernels(2*L+1, band)[direction]; c=L
    m=np.zeros_like(full)
    if direction==0: m[:, (c+1 if side>0 else 0):(2*L+1 if side>0 else c)]=1
    elif direction==1: m[(c+1 if side>0 else 0):(2*L+1 if side>0 else c), :]=1
    else:
        ii,jj=np.indices(full.shape); m=((ii>c) if side>0 else (ii<c)).astype(np.float32)
    return full*m

PERP={0:1,1:0,2:3,3:2}
def bridges(hw, printVis, k):
    """Print hidden under a pen stroke that CROSSES it: along the direction perpendicular to the
    pen stroke, print continues on BOTH sides of the pixel within about a pen width."""
    hwF=hw.astype(np.float32); pvF=printVis.astype(np.float32)
    L=2*k|1
    penScore=[]
    for kk in dir_kernels(L, BAND):
        penScore.append(cv2.filter2D(hwF,-1,kk,borderType=cv2.BORDER_CONSTANT)/kk.sum())
    penDir=np.argmax(np.stack(penScore),0)
    out=np.zeros(hw.shape,bool)
    reach=max(3,k)
    for d in range(4):
        a=half_kernels(reach,BAND,d,-1); b=half_kernels(reach,BAND,d,+1)
        sa=cv2.filter2D(pvF,-1,a,borderType=cv2.BORDER_CONSTANT); sb=cv2.filter2D(pvF,-1,b,borderType=cv2.BORDER_CONSTANT)
        both=(sa>=1.5)&(sb>=1.5)          # a few print pixels on each side, in line
        out|=both&(PERP[d]==penDir)       # only across the pen stroke, never along it
    return out&hw

def erase(img, hw, overlap, printLayer, odm):
    H,W=img.shape[:2]; k=max(5,int(max(H,W)/450))|1; s=k/9.0
    E=lambda d: cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(max(1,int(round(d)))|1,)*2)
    hw8=hw.astype(np.uint8)*255; pr8=printLayer.astype(np.uint8)*255
    printNear=cv2.dilate(pr8,np.ones((3,3),np.uint8))
    grown=cv2.dilate(hw8,E(13*s))
    area=hw8|(grown&((odm>0.04).astype(np.uint8)*255)&~printNear)
    area|=cv2.dilate(area,E(11*s))&~printNear
    ov=overlap.copy()
    if BRIDGE:
        k2=max(5,int(max(H,W)/450))|1
        ov|=bridges(hw, printLayer&~hw&(odm>0.3), k2)   # printLayer is confident print now
    # printed rules (long, thin, straight print runs) crossing the erased area are re-joined:
    # close short gaps along the line, keep only runs long AND thin (text lines are tall)
    pr_strong=(printLayer&~hw&(odm>0.3)).astype(np.uint8)*255
    L1=max(3,6*k)|1; L2=max(5,20*k)|1
    closedH=cv2.morphologyEx(pr_strong,cv2.MORPH_CLOSE,np.ones((1,L1),np.uint8))
    runs=cv2.morphologyEx(closedH,cv2.MORPH_OPEN,np.ones((1,L2),np.uint8))
    thick=cv2.morphologyEx(runs,cv2.MORPH_OPEN,np.ones((max(3,k)|1,1),np.uint8))
    rules=runs&~cv2.dilate(thick,np.ones((3,3),np.uint8))
    ov=ov|((rules>0)&(area>0))
    # leftover pen specks: small ink bits right beside the erased strokes that are not confident
    # print are part of the handwriting too
    nearHw=cv2.dilate(hw8,E(4*k+1))>0
    leftover=((odm>0.2)&nearHw&~printLayer&~hw).astype(np.uint8)
    nL,lL,stL,_=cv2.connectedComponentsWithStats(leftover,8)
    smallL=stL[:,4]<=(2*k)**2; smallL[0]=False
    area=area|(smallL[lL].astype(np.uint8)*255)
    restore=area&(ov.astype(np.uint8)*255)
    paper=area&~restore
    strong=(printLayer&~hw&(odm>0.3)).astype(np.float32); win=int(61*s)|1
    f=img.astype(np.float32)
    num=cv2.boxFilter(f*strong[...,None],-1,(win,win)); den=cv2.boxFilter(strong,-1,(win,win))
    fb=img[strong>0].mean(0) if strong.any() else np.array([40,40,40])
    local=np.where(den[...,None]>1e-3,num/np.maximum(den[...,None],1e-6),fb).astype(np.uint8)
    dst=img.copy(); dst[restore>0]=local[restore>0]
    # inpaint must sample PAPER only: print next to the erased area would otherwise bleed into it
    # (measured: the pen pixels left dark were all hw pixels beside print). Swap nearby print for
    # paper colour while inpainting, then put the real print (and restored overlap) back.
    sm=cv2.resize(img,None,fx=.125,fy=.125,interpolation=cv2.INTER_AREA)
    paperColor=cv2.GaussianBlur(cv2.resize(cv2.medianBlur(sm,21),(W,H)),(0,0),8)
    near=cv2.dilate(area,E(9*s))>0
    keep=(printNear>0)&near&(area==0) | (restore>0)
    tmp=dst.copy(); tmp[keep]=paperColor[keep]
    out=cv2.inpaint(tmp,paper,3,cv2.INPAINT_TELEA)
    out[keep]=dst[keep]
    return out

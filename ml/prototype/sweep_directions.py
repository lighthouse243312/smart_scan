import cv2, numpy as np, sys
S=sys.argv[1]; sys.path.insert(0,S); import refine as R
img=cv2.imread(f'{S}/synth_overlap.png'); gp,gh=np.load(f'{S}/synth_gt.npy'); go=gp&gh
coarse=np.load(f'{S}/s3_hw.npy'); ref=np.load(f'{S}/s3_ref.npy')
hwOnly=gh&~gp; prOnly=gp&~gh
for band in (5,):
  for lengths in ((2,4,6),):
    for dark in (1.3,1.35,1.4):
      R.BAND=band; R.LENGTHS=lengths; R.DARK=dark
      hw,ov,info=R.refine(img,coarse,ref)
      d=cv2.cvtColor(R.erase(img,hw,ov,info['printLayer'],info['odm']),cv2.COLOR_BGR2GRAY)
      e,p,o=(d[hwOnly]>=150).mean()*100,(d[prOnly]<128).mean()*100,(d[go]<128).mean()*100
      print(f'band={band} lengths={lengths} dark={dark}: pen erased {e:5.1f}% | print intact {p:5.1f}% | print under pen kept {o:5.1f}% | mean {(e+p+o)/3:5.1f}')

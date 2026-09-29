import cv2, numpy as np, sys
S=sys.argv[1]; sys.path.insert(0,S); from refine import refine, erase
img=cv2.imread(f'{S}/synth_overlap.png'); gp,gh=np.load(f'{S}/synth_gt.npy'); go=gp&gh
coarse=np.load(f'{S}/s3_hw.npy'); ref=np.load(f'{S}/s3_ref.npy')
hw,ov,info=refine(img,coarse,ref,np.load(f'{S}/s3_print.npy'),np.load(f'{S}/s3_frag.npy'))
dst=cv2.cvtColor(erase(img,hw,ov,info['printLayer'],info['odm']),cv2.COLOR_BGR2GRAY)
cv2.imwrite(f'{S}/r5_erased.png',dst)
old=cv2.imread(f'{S}/s3_erased.png',0)
hwOnly=gh&~gp; prOnly=gp&~gh
for nm,d in (('coarse v3 (current app)',old),('+ pixel refine & 4-dir',dst)):
    print(f'{nm:26s}: pen-only erased {(d[hwOnly]>=150).mean()*100:5.1f}% | print-only intact {(d[prOnly]<128).mean()*100:5.1f}% | print under pen kept {(d[go]<128).mean()*100:5.1f}%')

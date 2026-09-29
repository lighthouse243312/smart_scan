# Synthetic page with KNOWN overlap: print + pen strokes (blue-black, low saturation) crossing it.
import cv2, numpy as np, sys
sys.path.insert(0, sys.argv[1]); from directional import resolve
from PIL import Image, ImageDraw, ImageFont
S=sys.argv[1]
W,H=1400,800
doc=Image.new('L',(W,H),255); d=ImageDraw.Draw(doc)
f=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',30)
lines=["Họ và tên: Nguyễn Văn A      Lớp: 12A1","The quick brown fox jumps over the lazy dog.","Invoice total: 1,234,567 VND   Date: 12/09/2026","EDUCATION  COMPUTER ENGINEERING  HANOI UNIVERSITY","Chữ in bị viết đè bằng bút bi màu xanh đen."]
for i,l in enumerate(lines): d.text((60,60+i*140),l,fill=15,font=f)
d.line([(60,720),(1340,720)],fill=60,width=3)
doc=np.array(doc)
hw=Image.new('L',(W,H),255); h=ImageDraw.Draw(hw)
h.line([(40,90),(1300,70)],fill=0,width=4)
h.ellipse([420,190,700,270],outline=0,width=4)
h.line([(80,520),(300,470),(520,560),(760,470),(900,540)],fill=0,width=4,joint='curve')
h.text((700,330),"sai rồi!",fill=0,font=ImageFont.truetype('/System/Library/Fonts/Noteworthy.ttc',46))
h.line([(100,700),(1300,735)],fill=0,width=4)
for x in range(200,1300,150): h.line([(x,600),(x+40,660)],fill=0,width=4)
hw=np.array(hw)
ink=(255-hw).astype(np.float32)/255
color=np.array([60,70,120],np.float32)   # low-saturation blue-black ballpoint (RGB)
rgb=doc.astype(np.float32)[...,None].repeat(3,-1)
out=rgb*(255-(255-color)*ink[...,None])/255
img=cv2.cvtColor(out.astype(np.uint8),cv2.COLOR_RGB2BGR); img=cv2.GaussianBlur(img,(0,0),0.7)
noise=np.random.default_rng(0).normal(0,3,img.shape); img=np.clip(img+noise,0,255).astype(np.uint8)
cv2.imwrite(f'{S}/synth_overlap.png',img)
gt_print=doc<128; gt_hw=ink>0.3; gt_overlap=gt_print&gt_hw
np.save(f'{S}/synth_gt.npy',np.stack([gt_print,gt_hw]))
print('gt overlap px',gt_overlap.sum())

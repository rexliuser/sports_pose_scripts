import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
import argparse
import glob
import cv2
from tqdm import tqdm
import json
import functions as fn
import time

time1 = time.time()

parser = argparse.ArgumentParser(description='Run motionBERT')
parser.add_argument('projectname', metavar='FILENAME', type=str,help='projectname')
parser.add_argument('-r', action='store_true', help='rerun all, default False')
parser.add_argument('-f', action='store_true', help='flip left right, default False')
parser.add_argument('-c', help='crop video length in format MMSS-MMSS. source video must be in dataset/src/*.mp4', default='')
args = parser.parse_args()

codec = 'mp4v' #x264 #h264 #

connections = [(2,3),(5,6),(1,2),(4,5),(1,0),(4,0),(0,7),(7,8),(8,11),(8,14),(11,12),(12,13),(14,15),(15,16),(8,9),(9,10)]
viewrange = [(-40,20),(20,20)]
viewrange = [(-80,20),(-40,20)]

class Paths:
    def __init__(self, projectname):
        self.root = os.path.abspath('.')
        self.projects = os.path.join(self.root,'projects')
        self.alphapose = os.path.join(self.root,'AlphaPose')
        self.motionbert = os.path.join(self.root,'MotionBERT')
        self.projectname = os.path.join(self.projects,projectname)
        self.dataset = os.path.join(self.projectname,'dataset')
        self.projectalpha = os.path.join(self.projectname, 'alpha')
        self.projectmotion = os.path.join(self.projectname,'motion')
        self.npy = os.path.join(self.projectmotion, 'X3D.npy')
        self.animation = os.path.join(self.projectname, 'animation.mp4')
        self.video = os.path.join(self.projectname, 'video_cropped.mp4')
        self.video_combine = os.path.join(self.projectname, 'output_combine.mp4')
        self.alphavideo = os.path.join(self.projectalpha,'*.mp4')
        try:
            self.inputvideo = glob.glob(os.path.join(self.dataset,'*.mp4'))[0]
        except:
            if os.path.exists(os.path.join(self.dataset,'src')):
                path = glob.glob(os.path.join(self.dataset,'src','*.mp4'))[0]
                os.system(f'python extract_clip.py "{path}" {args.c} --out "{os.path.join(self.dataset,"video.mp4")}" 1> log.log 2>&1')
                self.inputvideo = os.path.join(self.dataset,"video.mp4")
                time.sleep(10)
            else:
                raise RuntimeError(f'Error: No mp4 video at {self.projectname}')
        if not os.path.exists(self.projectalpha): os.mkdir(self.projectalpha)
        if not os.path.exists(self.projectmotion): os.mkdir(self.projectmotion)
        if not os.path.exists(self.dataset): os.mkdir(self.dataset)
        

def update(frame):
    global points, pbar, validFrames, viewAngles, gridrange

    if frame in validFrames:
        pos = validFrames.index(frame)
    else:
        pos = 0
    
    pts = points[pos]
    ele = viewAngles[frame][1]
    azim = viewAngles[frame][0]

    ax.view_init(elev=ele, azim=azim)
    ax.lines.clear()
    ax.set_title(frame)

    if frame in validFrames:
        means = []
        for c in connections:
            means.append(np.mean([pts[c[0]][1],pts[c[1]][1]]))
        means = np.interp(np.array(means), (min(means),max(means)),(30,70))
        
        for i, c in enumerate(connections):
            u = means[i]
            pt = np.stack([pts[c[0]], pts[c[1]]],axis=0)
            color = (u/255,u*2/255,1,1)
            ax.plot(pt[:,0],pt[:,1],pt[:,2],c=color)
        
    pbar.update(1)

######################################
def preprocess():
    
    print('Proprocessing video (crop)...')
    vid = cv2.VideoCapture(paths.inputvideo)
    frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    res = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = vid.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    print(res,fps)
    
    out = cv2.VideoWriter(paths.video, fourcc, fps, (512,512))
    out.set(cv2.CAP_PROP_BITRATE, 100000)
    for i in tqdm(range(frames)):
        ret, img = vid.read()
        dim = min(img.shape[:2])
        l = (img.shape[1]-dim) // 2
        t = (img.shape[0]-dim) // 2
        r = l + dim
        b = t + dim
        img = img[t:b,l:r]
        img = cv2.resize(img, (512,512))
        if args.f: img = cv2.flip(img, 1)        
        out.write(img)
    vid.release()
    out.release()
    
    return fps

def runAlphaPose():
    print('Running Alphapose...')
    dir = os.getcwd()
    os.chdir(paths.alphapose)
    command =  f"python ./scripts/demo_inference.py --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth --video {os.path.join(paths.video)} --outdir {os.path.join(paths.projectalpha)} --detector yolo --save_video " #--pose_track
    os.system(command)
    os.chdir(dir)

def runMotionBert():
    print('Running MotionBERT...')
    dir = os.getcwd()
    os.chdir(paths.motionbert)
    command = f"python infer_wild.py --vid_path {paths.video} --json_path {os.path.join(paths.projectalpha,'alphapose-results_new.json')} --out_path {paths.projectmotion}"
    os.system(command)
    os.chdir(dir)

def processJson():
    print('Processing alphapose json...')
    path = os.path.join(paths.projectalpha,'alphapose-results.json')
    with open(path, 'r') as f:
        data = json.load(f)

    #idxs = {}
    #for d in data:
    #    if d['idx'] not in idxs.keys(): idxs[d['idx']] = d['score']
    #    else: idxs[d['idx']] += d['score']
    #
    #max_key = max(idxs, key=lambda x: idxs[x])
    #
    #names = []
    #for d in data:
    #    if d['idx'] == max_key:
    #        names.append(d)
    
    names = {}
    for d in data:
        if d['image_id'] in names:
            names[d['image_id']].append(d)
        else:
            names[d['image_id']] = [d]

    for key, lst in names.items():
        #scores = [(item['score'], item) for item in lst]
        scores = [(item['box'][2]*item['box'][3], item) for item in lst]
        scores = sorted(scores, key= lambda x:x[0], reverse=True)
        names[key] = scores[0][1]
    names = [dict for _, dict in names.items()]
    
    path = os.path.join(paths.projectalpha,'alphapose-results_new.json')
    with open(path, 'w') as f:
        json.dump(names, f)
        
    valid_frames = [dict['image_id'] for dict in names]
    valid_frames = [int(name.split('.')[0]) for name in valid_frames]
    return valid_frames

def stackvideo():
    print('combining video...')
    videopath = glob.glob(paths.alphavideo)[0]
    vid1 = cv2.VideoCapture(paths.animation)
    vid2 = cv2.VideoCapture(videopath)
    print(paths.animation)
    print(videopath)
    frames = int(vid1.get(cv2.CAP_PROP_FRAME_COUNT))
    
    res1 = (int(vid1.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vid1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    res2 = (int(vid2.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vid2.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = vid1.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    
    out = cv2.VideoWriter(paths.video_combine, fourcc, fps, (1024,512))
    out.set(cv2.CAP_PROP_BITRATE, 600000)
    for i in tqdm(range(frames)):
        ret, img1 = vid1.read()
        ret, img2 = vid2.read()
        img = cv2.hconcat([img1,img2])
        out.write(img)
    vid1.release()
    vid2.release()
    out.release()

def getViewAngles(frames, viewrange, fps):
    phase = 30*fps #sec, 2pi
    v = viewrange
    lst = []
    for i in range(frames):
        a = np.cos((i % phase)/phase *2*np.pi) * (v[1][0]-v[0][0])/2 + v[0][0] + (v[1][0]-v[0][0])/2
        b = np.cos((i % phase)/phase *2*np.pi) * (v[1][1]-v[0][1])/2 + v[0][1] + (v[1][1]-v[0][1])/2
        c = (a,b)
        lst.append(c)
    return lst

paths = Paths(args.projectname)

fps = 30
if not os.path.exists(paths.video) or args.r:
    fps = preprocess()
if not os.path.exists(os.path.join(paths.projectalpha,'alphapose-results_new.json')) or args.r:
    runAlphaPose()
validFrames = processJson()
if not os.path.exists(paths.npy) or args.r:
    runMotionBert()

print('Loading video...')
points = np.load(paths.npy)
print('Rectifying video...')
points = fn.preprocess(points, enable_movement=True)
#print('Fixing video...')
#points = fn.fix(points)

#points = points.tolist()
#points = [[np.dot(Rx,np.array(keypoint).T).T for keypoint in frame] for frame in points]
#points = np.array(points)
#
#print(points.shape)

#gridrange = np.min(points[:,:,x_]),np.max(points[:,:,x_]),np.min(points[:,:,y_]),np.max(points[:,:,y_]),np.min(points[:,:,z_]),np.max(points[:,:,z_])
gridrange = (-0.6,0.6,-0.6,0.6,-0.6,0.6)

fig = plt.figure()
fig.set_size_inches(512/100, 512/100)
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1,1,1])
mmin, mmax = min(gridrange[0],gridrange[2],gridrange[4]), max(gridrange[1],gridrange[3],gridrange[5])
ax.set_xlim3d(mmin,mmax)
ax.set_ylim3d(mmin,mmax)
ax.set_zlim3d(mmin,mmax)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.grid(color='gray', alpha=0.7)

frames = points.shape[0]
#frames = 30
print('Rendering pose video...')
viewAngles = getViewAngles(frames, viewrange, fps)
with tqdm(total=frames) as pbar:
    ani = FuncAnimation(fig, update, frames=frames, interval=fps)
    writer = FFMpegWriter(fps=30, bitrate=600)
    ani.save(paths.animation,writer=writer)

stackvideo()

print('Completed in {:.1f}s'.format(time.time()-time1))

#for i in range(17):
#    ax.text(points[0,i,0], points[0,i,1], points[0,i,2], str(i))
#
#pts = points[0]
#print(pts)
#for c in connections:
#    pt = np.stack([pts[c[0]], pts[c[1]]],axis=0)
#    ax.plot(pt[:,0],pt[:,1],pt[:,2],c='b')
#ax.scatter(pts[:,0],pts[:,1],pts[:,2])
#plt.show()
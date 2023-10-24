from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
import argparse
import glob
import cv2
import functions as fn
import numpy as np

codec = 'mp4v'

connections = [(2,3),(5,6),(1,2),(4,5),(1,0),(4,0),(0,7),(7,8),(8,11),(8,14),(11,12),(12,13),(14,15),(15,16),(8,9),(9,10)]


parser = argparse.ArgumentParser(description='Get Score')
parser.add_argument('projectname', metavar='FILENAME', type=str,help='projectname')
parser.add_argument('refprojectname', metavar='REFPROJ', type=str,help='reference projectname')
parser.add_argument('-r', action='store_true', help='rerun all, default False')
args = parser.parse_args()

class Paths:
    def __init__(self, projectname, ref=False):
        self.root = os.path.abspath('.')
        self.projects = os.path.join(self.root,'projects')
        self.projectname = os.path.join(self.projects,projectname)
        self.dataset = os.path.join(self.projectname,'dataset')
        self.projectalpha = os.path.join(self.projectname, 'alpha')
        self.scores = os.path.join(self.projectname,'scores')
        self.projectmotion = os.path.join(self.projectname,'motion')
        self.npy = os.path.join(self.projectmotion, 'X3D.npy')
        self.animation = os.path.join(self.scores, 'animation.mp4')
        self.video = os.path.join(self.projectname, 'video_cropped.mp4')
        self.video_combine = os.path.join(self.projectname, 'scores.mp4')
        self.alphavideo = os.path.join(self.projectalpha,'*.mp4')
        try:
            self.inputvideo = glob.glob(os.path.join(self.dataset,'*.mp4'))[0]
        except:
            raise RuntimeError(f'Error: No mp4 video at {self.projectname}')
        if not os.path.exists(self.projectalpha): os.mkdir(self.projectalpha)
        if not os.path.exists(self.projectmotion): os.mkdir(self.projectmotion)
        if not os.path.exists(self.dataset): os.mkdir(self.dataset)
        if not os.path.exists(self.scores) and not ref: os.mkdir(self.scores)

paths = Paths(args.projectname)
pointss0 = np.load(paths.npy)[:]
pointss = fn.preprocess(pointss0, enable_movement=True)
pointss_ = fn.preprocess(pointss0, enable_movement=False)

paths_ref = Paths(args.refprojectname, ref=True)
pointssref0 = np.load(paths_ref.npy)[:]
pointssref = fn.preprocess(pointssref0, enable_movement=True)
pointssref_ = fn.preprocess(pointssref0, enable_movement=False)

scores, stu_frames, teach_frames = fn.ScoreOfStudent(teacher=pointssref_, student=pointss_)
print(scores)
print(stu_frames)
print(teach_frames)


ele = 20
azim = -60

fig = plt.figure()
fig.set_size_inches(512/100, 512/100)
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1,1,1])
ax.set_xlim3d(-0.6,0.6)
ax.set_ylim3d(-0.6,0.6)
ax.set_zlim3d(-0.6,0.6)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

ax.set_ylabel('')
ax.set_zlabel('')
ax.grid(color='gray', alpha=0.7)
ax.view_init(elev=ele, azim=azim)

def getViewAngles(frames, viewrange, fps, phase=90):
    phase = phase #sec, 2pi
    v = viewrange
    lst = []
    for i in range(frames):
        a = np.cos((i % phase)/phase *2*np.pi) * (v[1][0]-v[0][0])/2 + v[0][0] + (v[1][0]-v[0][0])/2
        b = np.cos((i % phase)/phase *2*np.pi) * (v[1][1]-v[0][1])/2 + v[0][1] + (v[1][1]-v[0][1])/2
        c = (a,b)
        lst.append(c)
    return lst

def add_plot(ax, pts, color='b', ls='solid'):
    #means = []
    #for c in connections:
    #    means.append(np.mean([pts[c[0]][1],pts[c[1]][1]]))
    #means = np.interp(np.array(means), (min(means),max(means)),(30,70))
    
    for i, c in enumerate(connections):
        #u = means[i]
        pt = np.stack([pts[c[0]], pts[c[1]]],axis=0)
        #color = (u/255,u*2/255,1,1)
        ax.plot(pt[:,0],pt[:,1],pt[:,2],c=color, ls=ls)

def update1(f):
    global ax, pbar, frame, syned_points, viewAngles, scores, i, pointss, pointssref
    pts = pointss[frame]
    pts2 = syned_points

    ax.lines.clear()
    ax.set_title(f"Score: {scores[i]:.2f}")
    ax.set_xlabel(frame)
    
    ele = viewAngles[f][1]
    azim = viewAngles[f][0]
    ax.view_init(elev=ele, azim=azim)
    
    add_plot(ax,pts,'b')
    add_plot(ax,pts2,color='red', ls='--')
    
    #pbar.update(1)

def update2(f):
    global ax, pbar, frame, tframe, viewAngles, scores, i, pointss
    pts = pointss[f]

    ax.lines.clear()
    ax.set_title(f"")
    ax.set_xlabel(f)
    
    ele = viewAngles[f][1]
    azim = viewAngles[f][0]
    ax.view_init(elev=ele, azim=azim)
    
    add_plot(ax,pts,'b')
    
    pbar.update(1)

frames = pointss.shape[0]

duration=90
print(f'Rendering animations...')
for i in tqdm(range(len(scores))):
    if not os.path.exists(os.path.join(paths.scores,f'ani_{i}.mp4')) or args.r:
        frame = stu_frames[i]
        tframe = teach_frames[i]
        syned_points = fn.syncPoints(pointssref[tframe], pointss[frame])
        viewrange = [(-60,20),(-30,20)]
        viewAngles = getViewAngles(duration, viewrange, fps=30, phase=90)
        #with tqdm(total=duration) as pbar:
        pbar = 0
        ani = FuncAnimation(fig, update1, frames=duration, interval=30)
        writer = FFMpegWriter(fps=30, bitrate=1000)
        ani.save(os.path.join(paths.scores,f'ani_{i}.mp4'),writer=writer)

viewrange = [(-30,20),(-30,20)]
viewAngles = getViewAngles(frames, viewrange, fps=30, phase=frames)
print('Rendering main video...')
with tqdm(total=frames) as pbar:
    if not os.path.exists(os.path.join(paths.scores,f'animation_main.mp4')) or args.r:
        ani = FuncAnimation(fig, update2, frames=frames, interval=30)
        writer = FFMpegWriter(fps=30, bitrate=1000)
        ani.save(os.path.join(paths.scores,f'animation_main.mp4'),writer=writer)


#out = cv2.VideoWriter(os.path.join(paths.scores, f'output.mp4'), cv2.VideoWriter_fourcc(*'MJPG'), 30, (1024,1024))
v_animation_ = cv2.VideoCapture(os.path.join(paths.scores,f'animation_main.mp4'))
v_source = cv2.VideoCapture(paths.video)

fourcc = cv2.VideoWriter_fourcc(*codec)
out = cv2.VideoWriter(paths.video_combine, fourcc, 30, (1024,1024))
j=0
print('Stacking videos...')
for i in tqdm(range(frames)):
    
    #render main video
    ret, img1 = v_animation_.read()
    ret, img2 = v_source.read()
    img3 = np.ones((512,512,3), dtype=np.uint8)*255
    img4 = np.ones((512,512,3), dtype=np.uint8)*255
    out1 = cv2.hconcat([img1,img2])
    out2 = cv2.hconcat([img3,img4])
    out3 = cv2.vconcat([out1,out2])
    out.write(out3)

    if i in stu_frames:
        #render compare video
        v_freeze = cv2.VideoCapture(os.path.join(paths.scores,f'ani_{j}.mp4'))

        v_teach = cv2.VideoCapture(paths_ref.video)
        teacher_frame = teach_frames[j]
        total_frames = int(v_teach.get(cv2.CAP_PROP_FRAME_COUNT))
        for k in range(total_frames):
            ret, img_teach = v_teach.read()
            if k==teacher_frame:
                v_teach.release()
                break
        v_teach.release()

        
        for k in range(duration):
            ret, imgv = v_freeze.read()
            out1 = cv2.hconcat([imgv,img2])
            out2 = cv2.hconcat([img3,img_teach])
            out3 = cv2.vconcat([out1,out2])
            out.write(out3)
        
        v_freeze.release()
        j+=1
out.release()


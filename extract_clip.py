import argparse,os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import subprocess

parser = argparse.ArgumentParser(
    prog = 'extract_clip.py',
    description = 'Extract part of clip'
)
parser.add_argument('filepath', help='Input video path')
parser.add_argument('timestr', help='format:"0100-0130" in min, sec')
parser.add_argument('--out', help='output mp4 full path', default='')
args = parser.parse_args()

path=args.filepath
time1,time2=str(args.timestr).split('-')
m1=int(time1[:2])
s1=int(time1[2:4])
m2=int(time2[:2])
s2=int(time2[2:4])

file=os.path.basename(path).split('.')[0]
dir = os.path.dirname(path)
if args.out:
    output_file = args.out
else:
    output_file=os.path.join(dir,file+'-'+args.timestr+'.mp4')

start = str(m1*60+s1)
end = str(m2*60+s2)

commands = ['ffmpeg','-i',path,'-ss',start,'-to',end,'-q:v','0','-y',output_file]
print('command:',commands)
process = subprocess.Popen(commands)
#ffmpeg_extract_subclip(path,m1*60+s1,m2*60+s2, targetname=output_file)
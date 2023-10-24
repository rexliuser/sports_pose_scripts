import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

def rotate_x(points,rad):
    R = np.array([
        [1,0,0],
        [0,np.cos(rad),-np.sin(rad)],
        [0,np.sin(rad),np.cos(rad)] 
    ])
    points = np.dot(R, points.T).T
    return points

def rotate_y(points,rad):
    R = np.array([
        [np.cos(-rad), 0, np.sin(-rad)],
        [0, 1, 0],
        [-np.sin(-rad), 0, np.cos(-rad)]
    ])
    points = np.dot(R, points.T).T
    return points

def rotate_z(points,rad):
    #angle = np.radians(angle)
    R = np.array([
        [np.cos(rad), -np.sin(rad), 0],
        [np.sin(rad), np.cos(rad), 0],
        [0, 0, 1]
    ])
    points = np.dot(R, points.T).T
    return points

def inversesigmoid(x):
  return 1 / (1 + np.exp(x)) *2

def center_of_gravity(points):
    return np.mean(points, axis=0)

def moveToCenter(points):
    center = center_of_gravity(points)
    points -= center
    return points, center

def getPrincipleAxis(points):
    center = center_of_gravity(points)
    covariance = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    largest_eigenvalue_index = np.argmax(eigenvalues)
    smallest_eigenvalue_index = np.argmin(eigenvalues)
    axis_max = eigenvectors[:, largest_eigenvalue_index]
    axis_min = eigenvectors[:, smallest_eigenvalue_index]
    return axis_max, axis_min

def getPrincipleAxisMeanSeries(pointss):
    data = [getPrincipleAxis(points)[0] for points in pointss[...,:3]]
    data = [-d if d[2]<0 else d for d in data]
    mean = np.mean(data, axis=0)
    return mean

def correctRotation(points, axes=None):
    p0 = np.copy(points)
    
    if not axes: 
        axis_max, axis_min = getPrincipleAxis(points)
    else:
        axes = np.copy(axes)
        axis_max, axis_min = axes[0], axes[1]
    #print('aaa',axis_max)
        
    ang_x = -np.arctan2(axis_max[2],axis_max[1])+np.pi/2
    points = rotate_x(points, ang_x)
    axis_max = rotate_x(np.array([axis_max]), ang_x)[0]
    axis_min = rotate_x(np.array([axis_min]), ang_x)[0]
    ang_y = -np.arctan2(axis_max[2],axis_max[0])+np.pi/2
    points = rotate_y(points, ang_y)
    axis_max = rotate_y(np.array([axis_max]), ang_y)[0]
    axis_min = rotate_y(np.array([axis_min]), ang_y)[0]
    ang_z = -np.arctan2(axis_min[1],axis_min[0])
    points = rotate_z(points, ang_z)
    #axis_max = rotate_z(np.array([axis_max]), ang_z)[0]
    #axis_min = rotate_z(np.array([axis_min]), ang_z)[0]
    #print('bbb',axis_max)

    if points[10][2]<0:  #if head is overturned
        points = rotate_x(points, np.pi)
    if points[9][0] < points[8][0]: #if nose is behind neck:
        points = rotate_z(points, np.pi)
    
    p1 = np.copy(points)
    R = getRotationMatrix(p1,p0)
    
    return points, R

def correctSize(points):
    mean = getMeanStickSize(points)
    scale = 0.2 / mean
    points = points * scale
    return points, scale

def getMeanStickSize(points):
    connections = [(2,3),(5,6),(1,2),(4,5),(1,0),(4,0),(0,7),(7,8),(8,11),(8,14),(11,12),(12,13),(14,15),(15,16),(8,9),(9,10)]
    lengths = [np.linalg.norm(points[a]-points[b]) for a,b in connections]
    return np.mean(lengths)

def getRotationMatrix(p1,p2):
    r = Rotation.align_vectors(p1, p2)[0]
    return r.as_matrix()

def rectify(points, parameters=None, axes=None):
    if not parameters:
        points, T = moveToCenter(points)
        points, R = correctRotation(points, axes)
        points, S = correctSize(points)
    else:
        R, T, S = parameters
        points -= T
        points = np.dot(R, points.T).T
        points, S = correctSize(points)
    return points, [R, T, S]

def rectifySeries(pointss, ref_frame=-1, enable_movement = True):
    print('Rectifying series...')
    if enable_movement:
        i = pointss.shape[0]//2 if ref_frame <0 else ref_frame
        #max_axis = getFloorAxes(pointss)[:3]
        max_axis, _ = getPrincipleAxis(pointss[i])
        max_axis = getPrincipleAxisMeanSeries(pointss)
        min_axis = getPoseDirectionMeanSeries(pointss, max_axis)
        axes = [max_axis,min_axis]
        #print(max_axis)
        _, parameters = rectify(np.copy(pointss[i]), axes=axes)
        ptss = [rectify(point, parameters=parameters, axes=axes)[0] for point in tqdm(pointss)]
        pointss = np.array(ptss)
       
    else:
        pointss = np.array([rectify(point)[0] for point in tqdm(pointss)])
    return pointss

def addSpeedSeries(pointss):
    v = pointss[1:,...] - pointss[:-1,...]
    v = np.vstack(([v[0]],v))
    v = np.concatenate((pointss,v), axis=2)
    return v

def addAccSeries(pointss):
    v = pointss[1:,:,3:6] - pointss[:-1,:,3:6]
    v = np.vstack(([np.zeros(v[0].shape)],v))
    v = np.concatenate((pointss,v), axis=2)
    return v

def getScore(points1,points2):
    joint_dis = np.linalg.norm(points1[:,0:3]-points2[:,0:3], axis=1)
    joint_spe = np.linalg.norm(points1[:,3:6]-points2[:,3:6], axis=1)
    mean_dis = np.mean(joint_dis)
    mean_spe = np.mean(joint_spe)
    score_dis = inversesigmoid(mean_dis * 2)
    score_spe = inversesigmoid(mean_spe *25)
    #score_dis = mean_dis
    #score_spe = mean_spe
    score = score_dis*0.5 + score_spe*0.5
    return score

def compare(pointss1,pointss2):
    fm1, _ = getHitPoints(pointss1)
    fm2, _ = getHitPoints(pointss2)
    scores=[]
    for i in fm1:
        score = []
        for j in fm2:
            score.append(getScore(pointss1[i],pointss2[j]))
        scores.append(score)
    scores = np.array(scores)
    return scores, fm1, fm2

def ScoreOfStudent(teacher, student):
    scores, fm1, fm2 = compare(teacher,student)
    best_scores = np.max(scores,axis=0)
    best_teacher_frames = fm1[np.argmax(scores, axis=0)]
    return best_scores, fm2, best_teacher_frames

def fix(pointss):
    #fix every 243 frame from motion bert boundary issue
    aa = np.arange(0,pointss.shape[0],243)[1:]
    for a in aa:
        w = 5
        i,j = a-w//2,a+w//2+1
        ii,jj = i-2,j+2
        c = np.apply_along_axis(lambda m: np.convolve(m, np.ones(5), 'valid'), axis=0, arr=pointss[ii:jj,...]) / w
        pointss[i:j,...] = c
    
    #fix mid point teleport problem
    #b = pointss.shape[0]//2
    #pointss[b] = (pointss[b-1] + pointss[b+1]) /2
    return pointss

def bodyLength(points):
    body_sticks = {
        'head':(9,10),
        'neck':(8,9),
        'Rarm1':(8,14),
        'Rarm2':(14,15),
        'Rarm3':(15,16),
        'Larm1':(8,11),
        'Larm2':(11,12),
        'Larm3':(12,13),
        'Torso1':(8,7),
        'Torso2':(7,0),
        'Rleg1':(0,1),
        'Rleg2':(1,2),
        'Rleg3':(2,3),
        'Lleg1':(0,4),
        'Lleg2':(4,5),
        'Lleg3':(5,6)
    }
    ratio_dict = {}
    for key, pair in body_sticks.items():
        i,j = pair
        ratio_dict[key] = np.linalg.norm(points[i]-points[j])
    return np.array(list(ratio_dict.values())), ratio_dict

def getBodyLengthSeries(pointss):
    bodylength = []
    for points in pointss:
        bodylength.append(bodyLength(points)[0])
    return np.array(bodylength)

def getFastestJoint(pointss):
    speeds = pointss[10:,:,3:6]
    speedsv = np.linalg.norm(speeds, axis=2)
    meanv = np.mean(speedsv, axis=0)
    return np.argmax(meanv)

def getHitPoints(pointss):
    from scipy.signal import find_peaks
    
    j = getFastestJoint(pointss)
    series = np.linalg.norm(pointss[:,j,3:6], axis=1)
    series[:10] = 0
    
    period = getPeriod(series)
    
    peaks, _ = find_peaks(series, distance=max(period-30,30))
    series_peaks = series[peaks]
    pct = np.percentile(series, 90)
    frames = peaks[np.where(series_peaks>pct)]
    
    return frames, period

def getPeriod(series):
    import statsmodels.api as sm
    acf = sm.tsa.acf(series, nlags=300)
    acf[:10] = 0
    return np.argmax(acf[1:]) + 1

def preprocess(pointss, enable_movement=True):
    pointss = np.copy(pointss)
    pointss = rectifySeries(pointss, enable_movement=enable_movement)
    pointss = fix(pointss)
    pointss = addSpeedSeries(pointss)
    pointss = addAccSeries(pointss)
    return pointss

def getFloorAxes(pointss):
    import open3d as o3d
    pts = np.vstack((pointss[:,3,:3], pointss[:,6,:3]))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=5,
                                         num_iterations=5000)
    a,b,c,d = plane_model
    return a,b,c,d

def getPoseDirection(points, max_axis):
    points = np.copy(points[...,:3])
    shoulder_hips = points[[1,4,11,14]]
    _, _, vh = np.linalg.svd(shoulder_hips - np.mean(shoulder_hips, axis=0))
    v2 = np.array(max_axis)
    v1 = np.array(vh[-1])
    #v1 onto v2
    proj = ((np.dot(v1, v2)) / (np.dot(v2, v2))) * v2
    vector = v1 - proj
    u = vector / np.linalg.norm(vector)
    
    #check u dir align with head
    hd = points[9]-points[8]
    proj = ((np.dot(hd, u)) / (np.dot(u, u))) * u
    if np.dot(proj, u) < 0:
        u[:2] = -u[:2]
    return u
    
def getPoseDirectionMeanSeries(pointss, max_axis):
    data = [getPoseDirection(points, max_axis) for points in pointss[...,:3]]
    data = [-d if d[2]<0 else d for d in data]
    mean = np.mean(data, axis=0)
    return mean

def syncPoints(points,pointsref, offset=(0,-0.4,0)):
    points = np.copy(points[...,:3])
    pointsref = np.copy(pointsref[...,:3])
    center = center_of_gravity(points)
    centerref = center_of_gravity(pointsref)
    pts = points - center
    axisref = getPoseDirection(pointsref, (0,0,1))
    axis = getPoseDirection(points, (0,0,1))
    #print(axis, axisref)
    
    #ang = np.dot(axis, (0,1,0)) - np.dot(axisref,(0,1,0))
    ang = np.arctan2(axis[1],axis[0]) - np.arctan2(axisref[1],axisref[0])
    #print(ang*180/np.pi)
    pts = rotate_z(pts, -ang)
    
    pts += centerref + offset
    
    axis = getPoseDirection(points, (0,0,1))
    #print(axis, axisref)
    return pts
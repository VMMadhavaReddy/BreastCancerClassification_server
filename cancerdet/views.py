from django.shortcuts import render
import os

# Create your views here.
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from serverml.settings import BASE_DIR
image_dir = os.path.join(BASE_DIR,"static")
# ML CODE

import pickle as pk
import PIL
import numpy as np
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import convolve2d

from skimage.color import rgb2hsv

imgcgp=os.path.join(BASE_DIR,"cgp.jpg")

#feature extraction



def LBP33(image):
    imar=image
    newim=np.zeros((imar.shape),'uint64')
    for i in range(1,imar.shape[0]-1):
        for j in range(1,imar.shape[1]-1):
            s=[]
            s1=0
            s2=1
            for k in range(3):
                for l in range(3):
                    if not(k==l and l==1) and not(k==1 and l==0):
                        if k>1:
                            l11=2-l
                        else:
                            l11=l
                        if(imar[i-1+k][j-1+l11]>=imar[i][j]):
                            s1+=s2
                            s2=s2*2
                        else:
                            s2=s2*2
                    if k==1 and l==0:
                        if(imar[i-1+k][j-1+l]>=imar[i][j]):
                            s1+=2**7
            newim[i][j]=s1

    t=newim[1:newim.shape[0]-1,1:newim.shape[1]-1].flatten()
    
    t1=np.histogram(t.flatten(),range(257))[0]
    return t1
    
def LBPriv(image):
    imar=image
    l11=[0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 37, 39, 43, 45, 47, 51, 53, 55, 59, 61, 63, 85, 87, 91, 95, 111, 119, 127, 255]
    d={}
    for i in l11:
        d[i]=0
    newim=np.zeros((imar.shape),'uint64')
    print("s")
    for i in range(1,imar.shape[0]-1):
        for j in range(1,imar.shape[1]-1):
            s=[]
            s1=0
            s2=1
            for k in range(3):
                for l in range(3):
                    if not(k==l and l==1):
                        if k>=1:
                            l=2-l
                        if(imar[i-1+k][j-1+l]>=imar[i][j]):
                            s.append(1)
                        else:
                            s.append(0)
            temp=s[4]
            del s[4]
            s.append(temp)
            l=[]
            mn=300
            for k in range(8):
                s1=0
                for v in range(8):
                    s1+=s[7-v-k]*2**v
                if(s1<mn):
                    mn=s1
                    
            newim[i][j]=mn
    t=newim[1:newim.shape[0]-1,1:newim.shape[1]-1].flatten()    
    for i in t:
        d[i]=d[i]+1
    
    l12=[]
    for i in d.values():
        l12.append(i)
    print(2)
    t1=np.array(l12,dtype='uint64')    
    return t1

def CLBP(image,neighbors=8,radius=1,mode='h'):
    d_image=np.double(image)    
    spoints=np.zeros((neighbors,2))
    a = 2*np.pi/neighbors
    for i in range(neighbors):
        spoints[i][0] = -radius*np.sin((i)*a)
        spoints[i][1] = radius*np.cos((i)*a)
        print(np.round(-radius*np.sin((i)*a)))
        print(np.round(radius*np.cos((i)*a)))
    (ysize,xsize) = image.shape
    miny=np.min(spoints[:,0]);
    maxy=np.max(spoints[:,0]);
    minx=np.min(spoints[:,1]);
    maxx=np.max(spoints[:,1]);
    bsizey=int(np.ceil(np.max([maxy,0]))-np.floor(np.min([miny,0]))+1)
    bsizex=int(np.ceil(np.max([maxx,0]))-np.floor(np.min([minx,0]))+1)
    origy=int(-np.floor(np.min([miny,0])))
    origx=int(-np.floor(np.min([minx,0])))
    dx = int(xsize - bsizex)
    dy = int(ysize - bsizey)
    C = image[origy:origy+dy+1,origx:origx+dx+1];
    d_C = np.double(C);
    bins = 2**neighbors;
    CLBP_S=np.zeros((dy+1,dx+1));
    CLBP_M=np.zeros((dy+1,dx+1));
    CLBP_C=np.zeros((dy+1,dx+1));
    D=np.zeros((neighbors,dy+1,dx+1))
    Diff=np.zeros((neighbors,dy+1,dx+1))
    MeanDiff=np.zeros((neighbors,dy+1,dx+1))
    for i in range(neighbors):
        y = spoints[i,0]+origy
        x = spoints[i,1]+origx
        ry = int(np.round(y))
        rx = int(np.round(x))
        N = d_image[ry:ry+dy+1,rx:rx+dx+1]
        D[i] = np.array(N >= d_C,dtype='int64')   
        Diff[i] = np.abs(N-d_C)
        #D[i] = np.array(Diff[i]>= T,'int64')  
        DiffThreshold= np.mean(Diff)
    for i in range(neighbors):
        v = 2**(i)
        CLBP_S = CLBP_S + v*D[i]
        CLBP_M = CLBP_M + v*np.array(Diff[i]>=DiffThreshold,'int64');
    CLBP_C = np.array(d_C>=np.mean(d_image),'int64');
    CLBP_C = np.histogram(CLBP_C.flatten(),range(3))[0]
    print(CLBP_C)
    if (mode=='h' or mode=='hist' or mode=='nh'):
        CLBP_S=np.histogram(CLBP_S,range(bins+1))[0];
        CLBP_M=np.histogram(CLBP_M,range(bins+1))[0];
        if (mode=='nh'):
            CLBP_S=CLBP_S/np.sum(CLBP_S);
            CLBP_M=CLBP_M/np.sum(CLBP_M);
    t1=(CLBP_S,CLBP_M,CLBP_C)
    t=np.concatenate((t1[0],t1[1],t1[2]))
    return t


def lpq(img,winSize=3,decorr=1,freqestim=1,mode='h'):
    rho=0.90
    STFTalpha=1/winSize
    convmode='valid' 
    img=np.float64(img)
    r=(winSize-1)/2 
    x=np.arange(-r,r+1)[np.newaxis] 
    if freqestim==1:  
        w0=np.ones_like(x)
        w1=np.exp(-2*np.pi*x*STFTalpha*1j)
        w2=np.conj(w1)
    filterResp1=convolve2d(convolve2d(img,w0.T,convmode),w1,convmode)
    filterResp2=convolve2d(convolve2d(img,w1.T,convmode),w0,convmode)
    filterResp3=convolve2d(convolve2d(img,w1.T,convmode),w1,convmode)
    filterResp4=convolve2d(convolve2d(img,w1.T,convmode),w2,convmode)
    freqResp=np.dstack([filterResp1.real, filterResp1.imag,
                        filterResp2.real, filterResp2.imag,
                        filterResp3.real, filterResp3.imag,
                        filterResp4.real, filterResp4.imag])
    print(freqResp.shape)                    
    (freqRow,freqCol,freqNum)=freqResp.shape
    if decorr == 1:
        x1=np.arange(1,winSize+1)
        y1=np.arange(1,winSize+1)
        xp,yp=np.meshgrid(x1,y1)
        xp=xp.flatten()
        yp=yp.flatten()
        dd=np.zeros((winSize*winSize,winSize*winSize))
        for i in range(winSize*winSize):
            for j in range(winSize*winSize):
                dd[i][j]=np.sqrt((xp[i]-xp[j])*(xp[i]-xp[j])+(yp[i]-yp[j])*(yp[i]-yp[j]))
        C=np.power(rho,dd)
        q1=np.dot(w0.T,w1)
        q2=np.dot(w1.T,w0)
        q3=np.dot(w1.T,w1)
        q4=np.dot(w1.T,w2)
        s=q1.shape
        u1=q1.real.reshape(s[0]*s[1],1)
        u2=q1.imag.reshape(s[0]*s[1],1)
        u3=q2.real.reshape(s[0]*s[1],1)
        u4=q2.imag.reshape(s[0]*s[1],1)
        u5=q3.real.reshape(s[0]*s[1],1)
        u6=q3.imag.reshape(s[0]*s[1],1)
        u7=q4.real.reshape(s[0]*s[1],1)
        u8=q4.imag.reshape(s[0]*s[1],1)
        M=np.concatenate((u1.conj().transpose(),u2.conj().transpose(),u3.conj().transpose(),u4.conj().transpose(),u5.conj().transpose(),u6.conj().transpose(),u7.conj().transpose(),u8.conj().transpose()))
        print(M.shape,C.shape)      
        D=M.dot(np.matmul(C,M.conj().transpose()))
        U,S,V=np.linalg.svd(D)
        freqResp=np.reshape(freqResp,[freqRow*freqCol,freqNum])
        freqResp=(V.T.dot(freqResp.T)).T
        freqResp=np.reshape(freqResp,(freqRow,freqCol,freqNum))
    inds = np.arange(freqResp.shape[2])[np.newaxis,np.newaxis,:]
    LPQdesc=((freqResp>0)*(2**inds)).sum(2)
    if mode=='h':
        LPQdesc=np.histogram(LPQdesc.flatten(),range(257))[0]
    print(LPQdesc)
    return LPQdesc


def process_image(path):
    pil_image = PIL.Image.open(path).convert("L")
    imar=np.array(pil_image,"int64")
    t1=LBP33(imar)
    t2=LBPriv(imar)
    t3=CLBP(imar)
    t4=lpq(imar)
    t=[t1,t2,t3,t4]
    print("processing done")
    return t

def trainfeat(magn):
    allfeat={}
    allfeat["40X"]=["LBP331full","LBPriv1full","cLBPfull","LPQfull"]
    allfeat["100X"]=["LBP33full10","LBPRIVfull10","cLBPfull10","LPQfull10"]
    allfeat["200X"]=["LBP33full20","LBPRIV20full","CLBPfull20","LPQfull20"]
    allfeat["400X"]=["LBP33full40","LBPrivfull40","CLBPfull40","LPQfull40"]
    trf=[]
    for i in allfeat[magn]:
        tf=os.path.join(BASE_DIR,i)
        f=open(tf,'rb')
        trfi=pk.load(f)
        f.close()
        trf.append(trfi)
    print("training done")
    return trf



def sv(testimage,magn="40X"):
    cb=0
    cm=0
    cnt=0
    feat=['lbp','lbpriv','clbp','lpq']
    #par={}
    #par["40X"]=[['poly',2,0.1],['poly',10,2],['poly',1,0.09],['poly',5,0.5]]
    #par["100X"]=[['poly',10,0.1],['poly',1,7],['poly',0.5,0.1],['poly',5,0.1]]
    #par["200X"]=[['poly',4,0.1],['poly',1,7],['poly',1,0.1],['poly',0.005,0.9]]
    #par["400X"]=[['poly',1,0.1],['poly',0.5,3],['poly',0.5,0.1],['poly',0.009,0.9]]
    f=open(os.path.join(BASE_DIR,'DPclf'),'rb')
    Dclf=pk.load(f)
    f.close
    testfeature=process_image(testimage)
    da=trainfeat(magn)
    for i in range(4):
        (X,Y) =(da[i][0],da[i][1])
        X.append(testfeature[i])
        scaler = MinMaxScaler(feature_range=(0, 1))
        x1 = scaler.fit_transform(X)
        testfeature[i]=x1[-1]
        #del x1[-1]
        del X[-1]
        #k=par[magn][i][0]
        #C=par[magn][i][1]
        #g=par[magn][i][2]
        
        #clf=svm.SVC(kernel=k,gamma=g, C=C)
        #print(k,"gamma=",g,"c=",C,"not colored",feat[i])
        #clf.fit(x1[:-1],Y)
        clf=Dclf[magn][i]
        print(magn,":",i)
        predvalue=clf.predict([testfeature[i]])
        if 	predvalue[0]=='malignant':
            cm+=1
        else:
            cb+=1
        cnt+=1
    print("count:",cnt,cb)
    return cb
    #if cm>cb:
        #return "malignant"
    #else:
        #return "benign"
        
        
        
from mahotas.convolve import convolve
from mahotas.thresholding import otsu
from skimage.feature import greycomatrix
from skimage.feature.util import (FeatureDetector, DescriptorExtractor,
                            _mask_border_keypoints,
                            _prepare_grayscale_input_2D)

from skimage.feature import (corner_fast, corner_orientations, corner_peaks,
                       corner_harris)
from skimage.transform import pyramid_gaussian

from skimage.feature.orb_cy import _orb_loop
OFAST_MASK = np.zeros((31, 31))
OFAST_UMAX = [15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3]
for i in range(-15, 16):
    for j in range(-OFAST_UMAX[abs(i)], OFAST_UMAX[abs(i)] + 1):
        OFAST_MASK[15 + j, 15 + i] = 1
class GLCMf():
    def __init__(self,glcm):
        glcm=np.float64(glcm)
        size_glcm_1 = glcm.shape[0]
        size_glcm_2 = glcm.shape[1]
        size_glcm_3 = glcm.shape[2]
        #self.feat = []
        self.contr = np.zeros((1,size_glcm_3)).flatten()
        self.corrp = np.zeros((1,size_glcm_3)).flatten()
        self.energ = np.zeros((1,size_glcm_3)).flatten()
        self.entro = np.zeros((1,size_glcm_3)).flatten()
        self.homop = np.zeros((1,size_glcm_3)).flatten()
        self.sosvh = np.zeros((1,size_glcm_3)).flatten()
        self.savgh = np.zeros((1,size_glcm_3)).flatten()
        self.svarh = np.zeros((1,size_glcm_3)).flatten()
        self.senth = np.zeros((1,size_glcm_3)).flatten()
        self.dvarh2 = np.zeros((1,size_glcm_3)).flatten()
        self.denth = np.zeros((1,size_glcm_3)).flatten()
        self.inf1h = np.zeros((1,size_glcm_3)).flatten()
        self.inf2h = np.zeros((1,size_glcm_3)).flatten()
        #self.mxcch = np.zeros((1,size_glcm_3)).flatten()
        glcm_sum  = np.zeros((size_glcm_3,1)).flatten()
        glcm_mean = np.zeros((size_glcm_3,1)).flatten()
        glcm_var  = np.zeros((size_glcm_3,1)).flatten()
        u_x = np.zeros((size_glcm_3,1)).flatten()
        u_y = np.zeros((size_glcm_3,1)).flatten()
        s_x = np.zeros((size_glcm_3,1)).flatten()
        s_y = np.zeros((size_glcm_3,1)).flatten()
        p_x = np.zeros((size_glcm_1,size_glcm_3))
        p_y = np.zeros((size_glcm_2,size_glcm_3))
        p_xplusy = np.zeros(((size_glcm_1*2 - 1),size_glcm_3))
        p_xminusy = np.zeros(((size_glcm_1),size_glcm_3))
        hxy  = np.zeros((size_glcm_3,1)).flatten()
        hxy1 = np.zeros((size_glcm_3,1)).flatten()
        hx   = np.zeros((size_glcm_3,1)).flatten()
        hy   = np.zeros((size_glcm_3,1)).flatten()
        hxy2 = np.zeros((size_glcm_3,1)).flatten()
        #for k in range(size_glcm_3):
            #self.feat.append([])
        for k in range(size_glcm_3):
            glcm_sum[k] = sum(sum(glcm[:,:,k]))
            glcm[:,:,k] = glcm[:,:,k]/glcm_sum[k]
            glcm_mean[k] = np.mean(glcm[:,:,k])
            glcm_var[k] = (np.var(glcm[:,:,k]))
            #print(glcm_var[k])

            for i in range(size_glcm_1):
                for j in range(size_glcm_2):
                    self.contr[k] = self.contr[k] + (abs(i - j))**2*glcm[i,j,k]
                    self.energ[k] = self.energ[k] + (glcm[i,j,k]**2)
                    self.entro[k] = self.entro[k] - (glcm[i,j,k]*np.log(glcm[i,j,k] + np.spacing(1)))
                    self.homop[k] = self.homop[k] + (glcm[i,j,k]/( 1 + (i - j)**2))
                    self.sosvh[k] = self.sosvh[k] + glcm[i,j,k]*((i - glcm_mean[k])**2)
                    u_x[k]          = u_x[k] + (i)*glcm[i,j,k]
                    u_y[k]          = u_y[k] + (j)*glcm[i,j,k]
        for k in range(size_glcm_3):

            for i in range(size_glcm_1):

                for j in range(size_glcm_2):
                    p_x[i,k] = p_x[i,k] + glcm[i,j,k]
                    p_y[i,k] = p_y[i,k] + glcm[j,i,k]
                    p_xplusy[i+j,k] += glcm[i,j,k]
                    p_xminusy[abs(i-j),k] += glcm[i,j,k]
            u_x[k] = np.mean(p_x[:,k])
            u_y[k] = np.mean(p_y[:,k])
            s_x[k] = np.std(p_x[:,k])
            s_y[k] = np.std(p_y[:,k])
            #print(s_x)

        for k in range(size_glcm_3):

            for i in range(2*(size_glcm_1)-1):
                self.savgh[k] += (i+2)*p_xplusy[i,k]
                self.senth[k] -= (p_xplusy[i,k]*np.log(p_xplusy[i,k] + np.spacing(1)))

        for k in range(size_glcm_3):

            for i in range(2*(size_glcm_1)-1):
                self.svarh[k] += (((i+2) - self.senth[k])**2)*p_xplusy[i,k]

        for k in range(size_glcm_3):
            self.dvarh2[k] = np.var(p_xminusy[:,k])

            for i in range(size_glcm_1):
                self.denth[k] -= (p_xminusy[i,k]*np.log(p_xminusy[i,k] + np.spacing(1)))
        for k in range(size_glcm_3):
            hxy[k] = self.entro[k]
            for i in range(size_glcm_1):

                for j in range(size_glcm_2):
                    hxy1[k] = hxy1[k] - (glcm[i,j,k]*np.log(p_x[i,k]*p_y[j,k] + np.spacing(1)))
                    hxy2[k] = hxy2[k] - (p_x[i,k]*p_y[j,k]*np.log(p_x[i,k]*p_y[j,k] + np.spacing(1)))

                hx[k] = hx[k] - (p_x[i,k]*np.log(p_x[i,k] + np.spacing(1)))
                hy[k] = hy[k] - (p_y[i,k]*np.log(p_y[i,k] + np.spacing(1)))
            self.inf1h[k] = ( hxy[k] - hxy1[k] ) / ( np.max([hx[k],hy[k]]) )
            self.inf2h[k] = ( 1 - np.exp( -2*( hxy2[k] - hxy[k] ) ) )**0.5
            '''l = np.linalg.eig(Q[:,:,k])
            l= np.sort(l)
            l=l[::-1]
            self.mxcch[k] = l[k,1]**0.5'''

        corp = np.zeros((size_glcm_3,1)).flatten()
        for k in range(size_glcm_3):
            for i in range(size_glcm_1):
                for j in range(size_glcm_2):
                    corp[k] += ((i)*(j)*glcm[i,j,k])
            self.corrp[k] = (corp[k] - u_x[k]*u_y[k])/(s_x[k]*s_y[k])

class ORB(FeatureDetector, DescriptorExtractor):

    def __init__(self, downscale=1.2, n_scales=8,
                 n_keypoints=500, fast_n=9, fast_threshold=0.08,
                 harris_k=0.04):
        self.downscale = downscale
        self.n_scales = n_scales
        self.n_keypoints = n_keypoints
        self.fast_n = fast_n
        self.fast_threshold = fast_threshold
        self.harris_k = harris_k

        self.keypoints = None
        self.scales = None
        self.responses = None
        self.orientations = None
        self.descriptors = None

    def _build_pyramid(self, image):
        image = _prepare_grayscale_input_2D(image)
        return list(pyramid_gaussian(image, self.n_scales - 1,
                                     self.downscale))

    def _detect_octave(self, octave_image):
        # Extract keypoints for current octave
        fast_response = corner_fast(octave_image, self.fast_n,
                                    self.fast_threshold)
        keypoints = corner_peaks(fast_response, min_distance=1)

        if len(keypoints) == 0:
            return (np.zeros((0, 2), dtype=np.double),
                    np.zeros((0, ), dtype=np.double),
                    np.zeros((0, ), dtype=np.double))

        mask = _mask_border_keypoints(octave_image.shape, keypoints,
                                      distance=16)
        keypoints = keypoints[mask]

        orientations = corner_orientations(octave_image, keypoints,
                                           OFAST_MASK)

        harris_response = corner_harris(octave_image, method='k',
                                        k=self.harris_k)
        responses = harris_response[keypoints[:, 0], keypoints[:, 1]]

        return keypoints, orientations, responses

    def _extract_octave(self, octave_image, keypoints, orientations):
        mask = _mask_border_keypoints(octave_image.shape, keypoints,
                                      distance=20)
        keypoints = np.array(keypoints[mask], dtype=np.intp, order='C',
                             copy=False)
        orientations = np.array(orientations[mask], dtype=np.double, order='C',
                                copy=False)

        descriptors = _orb_loop(octave_image, keypoints, orientations)

        return descriptors, mask


    def detect_and_extract(self, image):
        pyramid = self._build_pyramid(image)

        keypoints_list = []
        responses_list = []
        scales_list = []
        orientations_list = []
        descriptors_list = []

        for octave in range(len(pyramid)):

            octave_image = np.ascontiguousarray(pyramid[octave])

            keypoints, orientations, responses = \
                self._detect_octave(octave_image)

            if len(keypoints) == 0:
                keypoints_list.append(keypoints)
                responses_list.append(responses)
                descriptors_list.append(np.zeros((0, 256), dtype=np.bool))
                continue

            descriptors, mask = self._extract_octave(octave_image, keypoints,
                                                     orientations)

            keypoints_list.append(keypoints[mask] * self.downscale ** octave)
            responses_list.append(responses[mask])
            orientations_list.append(orientations[mask])
            scales_list.append(self.downscale ** octave *
                               np.ones(keypoints.shape[0], dtype=np.intp))
            descriptors_list.append(descriptors)
        keypoints = np.vstack(keypoints_list)
        responses = np.hstack(responses_list)
        scales = np.hstack(scales_list)
        orientations = np.hstack(orientations_list)
        descriptors = np.vstack(descriptors_list).view(np.bool)

        if keypoints.shape[0] < self.n_keypoints:
            self.keypoints = keypoints
            self.scales = scales
            self.orientations = orientations
            self.responses = responses
            self.descriptors = descriptors
        else:
            # Choose best n_keypoints according to Harris corner response
            best_indices = responses.argsort()[::-1][:self.n_keypoints]
            self.keypoints = keypoints[best_indices]
            self.scales = scales[best_indices]
            self.orientations = orientations[best_indices]
            self.responses = responses[best_indices]
            self.descriptors = descriptors[best_indices]
def tas(img, thresh, margin):
    M = np.ones((3, 3))
    M[1, 1] = 10
    bins = np.arange(11)
    def cmp(img):
        V = convolve(img.astype(np.uint8), M)
        values,_ = np.histogram(V, bins=bins)
        values = values[:9]
        s = values.sum()
        if s > 0:
            return values/float(s)
        return values

    def comp(bimg):
        at.append(cmp(bimg))
        ant.append(cmp(~bimg))

    at = []
    ant = []
    mu = img[img > thresh].sum() / (np.sum(img > thresh) + 1e-8)
    comp( (img > mu - margin) * (img < mu + margin) )
    comp(img > mu - margin)
    comp(img > mu)
    return np.concatenate(at + ant)
def pftas(img):
    T = otsu(img)
    pixels = img[img > T]
    std = pixels.std()
    Pftas=tas(img, T, std) 
    return Pftas
Dic={}    
def fun(a , b):
    cb1=0
    cm1=0
    pil_image = PIL.Image.open(a).convert("L")
    a=np.array(pil_image)
    L=['400X', '100X', '200X', '40X']
    i=0
    
    k=[['rbf','rbf','rbf','rbf'],['rbf','rbf','rbf','rbf'],['poly','poly','poly','poly']]
    
    g=[[20.05,20.5,20.5,20.05],[20.05,20.5,20.5,20.05],[1.5,1.5,1.5,1.5]]
    c=[[16,16,16,16],[16,16,16,16],[50,50,50,50]]
    d=[[3,3,3,3],[3,3,4,4],[4,4,4,4]]
    Res=[]
    Mdl=['glcm','orb','pftas','','','','']
    feat=[[],[],[]]
    
    orb = ORB(n_keypoints=20)
    orb.detect_and_extract(a)
    Orb=orb.descriptors
    Orb=[len(i[i==True]) for i in np.array(Orb)]

    mtx=greycomatrix(a,[1,2],[0,np.pi/4,np.pi/2,np.pi*3/4])
    Mtx=mtx[:,:,0,:]
    glcmf=GLCMf(Mtx)
    glcmff=[np.mean(glcmf.contr),np.mean(glcmf.corrp),np.mean(glcmf.energ),np.mean(glcmf.entro),np.mean(glcmf.homop),np.mean(glcmf.sosvh),np.mean(glcmf.savgh),np.mean(glcmf.svarh),np.mean(glcmf.senth),np.mean(glcmf.dvarh2),np.mean(glcmf.denth),np.mean(glcmf.inf1h),np.mean(glcmf.inf2h)]                               
    f=open(os.path.join(BASE_DIR,"Pglcmscale"),'rb')         
    scl=pk.load(f)
    f.close()
    kk=np.array(glcmff)/scl
    
    feat[2].append(pftas(a))
    feat[0].append(kk)
    feat[1].append(Orb)
    print(feat)
    
    f=open(os.path.join(BASE_DIR,'Pclf'),'rb')
    Dic=pk.load(f)
    f.close()
    for j in range(3):
        i=0
        #f=open('P'+Mdl[j],'rb')
        #X,Y=pk.load(f)
        #f.close()
        for l in L:
            if l==b:
            #print(k[0][0])
            #Dic[Mdl[j]+l]=svm.SVC(kernel=k[j][i],gamma=g[j][i] , C=c[j][i],degree=d[j][i])
            #print(X['40X'][0])
            #Dic[Mdl[j]+l].fit(X[l],Y[l])
                #print(i)
                l=Dic[Mdl[j]+l].predict(feat[j])
                if l==0:
                    cb1+=1
                    print('b')
                else:
                    cm1+=1
            #Res.append(l)  
            #print(Res)
            i+=1
    return cb1 
    #print(Res)    
#fun(img,'40X')
#print(Dic)             
                   
def bound(im):
    pil_image = PIL.Image.open(im)
    rgb_img=np.array(pil_image)
    hsv_img = rgb2hsv(rgb_img)
    hue_img = hsv_img[:, :, 0]
    value_img = hsv_img[:, :, 2]
    if hue_img.mean()< 0.5:
        return 1
    return 0        
                             

def index(request):
    return render(request,'home.html')


def test(request):
    t='a'
    if request.method=="POST":
        magn=request.POST.get('Magnification')
        if magn=='Select magnification':
            return HttpResponse('<h1>Please select magnification value.</h1><a href="/../" style="position:absolute; right:1%"><button>Home</button></a>')
        else:
            try :
                uploaded_file=request.FILES['document']
            except:
                return HttpResponse('<h1>Please choose a file.</h1><a href="/../" style="position:absolute; right:1%"><button>Home</button></a>')
            filename=uploaded_file.name
            fs=FileSystemStorage()
            fs.save(os.path.join(image_dir,uploaded_file.name),uploaded_file)
            imagePath = os.path.join("static/",filename)
            try :
                pil_image=PIL.Image.open(imagePath).convert('L')
            except :
                return HttpResponse('<h1>Please select appropriate image.</h1><a href="/../" style="position:absolute; right:1%"><button>Home</button></a>')
            if bound(imagePath):
                return HttpResponse('<h1>Please select appropriate image.</h1><a href="/../" style="position:absolute; right:1%"><button>Home</button></a>')
            c1=sv(imagePath,magn)
            c2=fun(imagePath,magn)
            print('final cb:',c1+c2)
            if c1+c2>=4:
                t='Benign'
            else:
                t='Malignant'
            print(imagePath)
            return render(request,'test.html',{'testresult':t,'test_image':imagePath})
    return render(request,'test.html',{'testresult':t})


def train(request):
    t='a'
    img=dict()
    if request.method=="POST":
        magn=request.POST.get('Magnification')
        t=request.POST.get('reslt')
        if magn=='Select magnification':
            return HttpResponse('<h1>Please select magnification value.</h1><a href="/../" style="position:absolute; right:1%"><button class="bun3"  type="button" style = "background-color: rgba(143, 170, 206,0.5)">Back</button></a>')
        else:
            try :
                uploaded_file=request.FILES['document']
            except:
                return HttpResponse('<h1>Please choose a file.</h1><a href="/../" style="position:absolute; right:1%"><button>Home</button></a>')
            filename=uploaded_file.name
            fs=FileSystemStorage()
            fs.save(os.path.join(image_dir,uploaded_file.name),uploaded_file)
            imagePath = os.path.join("static/",filename)
            try :
                pil_image=PIL.Image.open(imagePath).convert('L')
            except :
                return HttpResponse('<h1>Please select appropriate image.</h1><a href="/../" style="position:absolute; right:1%"><button>Home</button></a>')
            if bound(imagePath):
                return HttpResponse('<h1>Please select appropriate image.</h1><a href="/../" style="position:absolute; right:1%"><button>Home</button></a>')
            #t=sv(imagePath,magn)  write code for training
            f=open(os.path.join(BASE_DIR,'tranfeat'),'rb')
            img=pk.load(f)
            f.close()
            l0=magn
            l1=process_image(imagePath)
            l2=t
            d1=(l0,l1,l2)
            img[filename]=d1
            f=open(os.path.join(BASE_DIR,'tranfeat'),'wb')
            pk.dump(img,f)
            f.close()
            print(imagePath)
            return render(request,'train.html',{'testresult':t,'test_image':imagePath})
    return render(request,'train.html',{'testresult':t})





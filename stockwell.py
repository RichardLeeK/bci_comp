import numpy as np
from scipy import signal 
import sys
import copy
import math
from obspy.core import Trace

def st(data,minfreq=0,maxfreq=None,samprate=None,freqsamprate=1,remove_edge=False,analytic_signal=False,factor=1):
    if data.shape[0] <= 1 or len(data.shape) > 1 :
        raise TypeError('input data invalid ,please check!') 
    if not maxfreq and not samprate:
        #regard signal as 1 second length
        maxfreq=len(data)//2
        samprate=len(data)
    if maxfreq and not samprate:
        samprate=len(data)
    if not maxfreq and samprate:
        maxfreq=samprate//2

    orig=copy.copy(data)
    st_res=np.zeros((int((maxfreq-minfreq)/freqsamprate)+1,len(data)),dtype='c8')	
    
    if remove_edge:
        tmp=Trace(data=orig)
        tmp.detrend('polynomial',order=2)
        tmp.taper(0.04)
        orig=tmp.data
    if analytic_signal:
        print('analytic_signal selected;  Calculating analytic signal!')
        orig=signal.hilbert(orig)

    vec=np.hstack((np.fft.fft(orig),np.fft.fft(orig)))

    if minfreq == 0:
        st_res[0]=np.mean(orig)*np.ones(len(data))
    else:
        st_res[0]=np.fft.ifft(vec[minfreq:minfreq+len(data)]*g_window(len(data),minfreq,factor))

    for i in range(freqsamprate,(maxfreq-minfreq)+1,freqsamprate):
        st_res[int(i/freqsamprate)]=np.fft.ifft(vec[minfreq+i:minfreq+i+len(data)]*g_window(len(data),minfreq+i,factor))
    return st_res


def ist(st_matrix):
    # 1-D inverse stockwell transform code modified by GaoSong from origion ist.m code
	#    the input matrix must be redundant size(N,N//2+1)
    stsp=np.sum(st_matrix,axis=1)
    if st_matrix.shape[1] % 2 != 0:
        negsp=stsp[2:].T[::-1]
    else:
        negsp=stsp[2:-1].T[::-1]

    fullstsp=np.hstack((np.conjugate(stsp.T),negsp))
    ts=np.fft.ifft(fullstsp).real
    return ts

def g_window(length,freq,factor):    
    gauss=signal.gaussian(length,std=(freq)/(2*np.pi*factor))
    gauss=np.hstack((gauss,gauss))[length//2:length//2+length]
    return gauss



def fdost(data,origion=False):
    # return  DOST  coefficient , first positive frequency after negative(same as fft)
    if not origion: 
        N=len(data)
        fdata=np.fft.fft(data)
        p_limit=math.floor(math.log2(N))
        res=[]
        ns=0
        for i in range(p_limit):
            _, beta, _=vbt(i,N) # mu, beta, tau
            r_matrix=np.array([])
            for j in range(beta):
                r_matrix=np.hstack((r_matrix,np.array((-1)**j)))
            
            v_matrix=r_matrix[:,np.newaxis].dot(np.fft.ifft(fdata[ns:ns+beta]).reshape(1,beta))*np.sqrt(beta)
            ns+=beta
            res.append(v_matrix)
        for i in range(-p_limit,0):
            _, beta, _=vbt(i,N) # mu, beta, tau
            r_matrix=np.array([])
            for j in range(beta):
                r_matrix=np.hstack((r_matrix,np.array((-1)**j)))
            
            v_matrix=r_matrix[:,np.newaxis].dot(np.fft.ifft(fdata[ns:ns+beta]).reshape(1,beta))*np.sqrt(beta)
            ns+=beta
            res.append(v_matrix)
        return res

    # return total redundant array in positive frequency
    elif origion:
        N=len(data)
        fdata=np.fft.fft(data)
        p_limit=math.floor(math.log2(N))
        ns=0
        for i in range(p_limit):
            _, beta, _=vbt(i,N) # mu, beta, tau
            r_matrix=np.array([])
            for j in range(beta):
                r_matrix=np.hstack((r_matrix,np.array((-1)**j)))
            v_matrix=r_matrix[:,np.newaxis].dot(np.fft.ifft(fdata[ns:ns+beta]).reshape(1,beta))*np.sqrt(beta)
            ns+=beta
            if i == 0 :
                fdost_res=np.tile(v_matrix,N)
            elif i == 1 :
                fdost_res=np.vstack((fdost_res,np.tile(v_matrix,N)))
            else :
                for m in range(beta):
                    tmp=np.array([])
                    for n in range(beta):
                        tmp=np.hstack((tmp,np.tile(np.asarray(v_matrix[m][n]),int(N/beta))))
                        
                    fdost_res=np.vstack((fdost_res,tmp))
                
        return fdost_res

def vbt(p,N):
    # compute orthonormal basis vector parrameter mu,beta,tau
    if p == 0:
        mu=0;beta=1;tau=0
    elif p == 1:
        mu=1;beta=1;tau=0
    elif p > 1:
        mu=2**(p-1)+2**(p-2)
        beta=2**(p-1)
        tau=list(range(beta-1))

    elif p == -1:
        mu=-1;beta=1;tau=0
    elif p == -math.floor(math.log2(N)):
        mu=-2**(-p-1);beta=1;tau=0
    else:
        mu=-2**(-p-1)-2**(-p-2)+1
        beta=2**(-p-1)
        tau=list(range(beta-1))
    return mu,beta,tau


def idost(fdost_res):
    N=2**(len(fdost_res)/2)
    p_limit=math.floor(math.log2(N))
    ori=np.array([])
    for i in range(p_limit):
        _, beta, _=vbt(i,N) # mu, beta, tau
        r_matrix=np.array([])
        for j in range(beta):
            r_matrix=np.hstack((r_matrix,np.array((-1)**j)))
        iff=np.linalg.pinv(r_matrix[:,np.newaxis]).dot(fdost_res[i]/np.sqrt(beta)).flatten()
        ff=np.fft.fft(iff)
        ori=np.hstack((ori,ff))
    for i in range(-p_limit,0):
        _, beta, _=vbt(i,N) # mu, beta, tau
        r_matrix=np.array([])
        for j in range(beta):
            r_matrix=np.hstack((r_matrix,np.array((-1)**j)))
        iff=np.linalg.pinv(r_matrix[:,np.newaxis]).dot(fdost_res[i+len(fdost_res)]/np.sqrt(beta)).flatten()
        ff=np.fft.fft(iff)
        ori=np.hstack((ori,ff))
    ori=np.fft.ifft(ori).real
    return ori


def sf_partition(N):
    # return spatial domain and frequency domain refer to fdost(data)
    p_limit=math.floor(math.log2(N))
    sf=[]
    ff=[]
    freq=[x for x in range(N//2)]+[y for y in range(-N//2,0)]
    k=0
    for i in range(p_limit):
        _, beta, _=vbt(i,N) # mu, beta, tau        
        for j in range(beta):
            ss=[N/(2*beta)*i for i in range(2*beta) if i %2 == 1]
            if j == 0:
                v_matrix=np.array(ss)
                h_matrix=np.array(np.tile(np.array(freq[k]),beta))
            else:
                v_matrix=np.vstack((v_matrix,np.array(ss)))       
                h_matrix=np.vstack((h_matrix,np.tile(np.array(freq[k]),beta)))
            k+=1
        sf.append(v_matrix)
        ff.append(h_matrix)
    for i in range(-p_limit,0):
        _, beta, _=vbt(i,N) # mu, beta, tau       
        for j in range(beta):
            ss=[N/(2*beta)*i for i in range(2*beta) if i %2 == 1]
            if j == 0:
                v_matrix=np.array(ss)
                h_matrix=np.array(np.tile(np.array(freq[k]),beta))
            else:
                v_matrix=np.vstack((v_matrix,np.array(ss)))      
                h_matrix=np.vstack((h_matrix,np.tile(np.array(freq[k]),beta)))
            k+=1
        sf.append(v_matrix)
        ff.append(h_matrix)
    return sf,ff

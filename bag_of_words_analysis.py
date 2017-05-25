import numpy as np
import cv2
import os
import social_force_calculation
from sklearn.cluster import KMeans
import lda
import lda.datasets

'''
This is the reproduction of "Abnormal Crowd Behavior Detection using Social Force Model" by Ramin Mehran, Alexis Oyama, Mubarak Shah
The document is attached to project

bag_of_words_analysis.py includes functionality of getting codewords from videos, doing kmeans clusterization and doing
Latent Dirichlet Allocation

References

1. "Abnormal Crowd Behavior Detection using Social Force Model" by Ramin Mehran, Alexis Oyama, Mubarak Shah

'''


'''
K_MEANS this method gets all visual words created earlier from directory and did k means clustering
and saves cluster centers to dict.npy

Arguments:
    clusterSize - cluster count for k means analysis
    n - visual word height and width in our case n=5 Reference 1 page 6
    T - visual word depth in our case T=10 frames Reference 1 page 6
Returns:
    kmeans -  object created by sklearn.cluster for k mean clusterization
'''

def k_means(clusterSize,n=5,T=10):
    points=np.empty((0,n*n*T*3))
    visualwords = os.listdir("Visual Words")
    for word in visualwords:
        data = np.load('Visual Words/' + word)
        for i in range(data.shape[0]):
            points=np.append(points,np.array([np.float32(data[i].flatten())]),axis=0)
    kmeans = KMeans(n_clusters=clusterSize, random_state=0).fit(points)
    np.save('Dictionary/kmeans.npy', kmeans.cluster_centers_)
    return kmeans


'''
LATENT_DIRICHLET_ALLOCATION this method calculates log likelihood from corpus
and saves likelihoods

Arguments:
    size - topics size
Returns:
    saves likelihoods for every T frames of test video
'''
def latent_dirichlet_allocation(size):
    model = lda.LDA(n_topics=size, n_iter=500, random_state=1)
    visualwords = os.listdir("LDA")
    likelihood=np.array([])
    for word in visualwords:
        data = np.load('LDA/' + word)
        print data.shape
        print word
        for i in range(data.shape[0]):
            model.fit(data[i].reshape(11,15).astype(np.int32))
            likelihood=np.append(likelihood,model.loglikelihood())
    np.save('LDAresult/likelihood.npy', likelihood)


'''
IS_FORCE_EXIST this method gets visual word and checks if it has forces or not
this is to avoid visual words with zero forces

Arguments:
    flow - visual word of shape nxnxT (in our analysis we used 5x5x10)
Returns:
    boolean - true if there are some forces, false if not
'''
def is_force_exist(flow):
    d=flow[np.where(flow>0.1)] # 0.1 threshold to avoid very small forces maybe needs to increase
    if len(d)>0:
        return True
    return False


'''
GET_VISUAL_WORDS_AND_SAVE this method gets visual words from force flows

Arguments:
    directory - directory of videos (for dataset training only videos with normal crowd behavior were taken see Reference 1 page 4
    n - visual word height and width in our case n=5 Reference 1 page 6
    T - visual word depth in our case T=10 frames Reference 1 page 6
    C - codebook size(cluster count) C=10 Reference 1 page 6
    K - number of visual words extracted from every T frames K=30 Reference 1 page 6
    resize - video resize percent from initial image for calculation simplicity 25% is taken
Returns:
    creates array with visual words for every video and save it in Visual Words directory
'''

def get_visual_words_and_save(directory,n=5,T=10,C=10,K=30,resize=0.25):
    videos = os.listdir(directory)
    for file in videos:
        print file
        fn = file.split(".")[0]
        Words = np.empty((0, T, n, n, 3))
        IsRead,Force, flow = social_force_calculation.force_calculation(directory+"/" + file, 0.5, 0, resize)
        if not IsRead:
            continue
        frCount = Force.shape[0]
        w = np.array([x for x in xrange(0, Force.shape[1] - n, n)])
        h = np.array([x for x in xrange(0, Force.shape[2] - n, n)])
        rng=frCount / (T-1)
        if frCount % (T-1)==0:
            rng-=1
        for i in range(rng):
            F1 = Force[T * i-i:T * i + T-i, :, :, :]  # getting one clip of 10 frames with 1 frame overlap
            Flow1 = flow[T * i-i:T * i + T-i, :, :]
            np.random.shuffle(w)  # random shuffle of array for visual words random selection
            np.random.shuffle(h)
            j = 0
            br = False
            for k in range(w.shape[0]):
                for d in range(h.shape[0]):
                    Word = F1[:, w[k]:w[k] + n, h[d]:h[d] + n, :]
                    if is_force_exist(Flow1[:, w[k]:w[k] + n, h[d]:h[d] + n]):
                        Words = np.append(Words, np.array([Word]), axis=0)
                        j += 1
                    if j >= K:
                        br = True
                        break
                if br:
                    break
        np.save('Visual Words/visualWords_' + fn + '.npy', Words)


'''
CREATE_LDA_MODEL_FROM_VISUAL_WORDS this method creates matrix of video where where the
nxnxT submatrixes encoded with single word 1-10 that shows to what cluster it belongs

Arguments:
    directory - directory of videos 
    n - visual word height and width in our case n=5 Reference 1 page 6
    T - visual word depth in our case T=10 frames Reference 1 page 6
    kmeans - object created by sklearn.cluster for k mean clusterization
    resize - video resize percent from initial image for calculation simplicity 25% is taken
Returns:
    creates array for Latent Dirischlet analysis and saves it to LDA directory
'''
def create_lda_model_from_visual_words(kmeans,directory,n=5,T=10,resize=0.25):
    videos = os.listdir(directory)
    for file in videos:
        fn = file.split(".")[0]
        print file
        IsRead,Force, _ = social_force_calculation.force_calculation(directory+"/" + file, 0.5, 0, resize)
        if not IsRead:
            continue
        frCount = Force.shape[0]
        dim=((Force.shape[1]-n)/n+1)*((Force.shape[2]-n)/n+1)
        ar1=np.empty((0,dim))
        for i in range(frCount / T):
            F1 = Force[T * i:T * i + T, :, :, :]  # getting one clip of 10 frames
            arr=np.array([])
            for k in xrange(0, Force.shape[1] - n, n):
                for d in  xrange(0, Force.shape[2] - n, n):
                    Word = F1[:, k:k + n, d:d + n, :]
                    cluster=kmeans.predict(np.array([Word.flatten()]))
                    arr=np.append(arr,cluster[0])
            ar1=np.append(ar1,np.array([arr.copy()]),axis=0)
        np.save('LDA/set_' + fn + '.npy', ar1)


if __name__ == '__main__':
    import sys

    visualWords=False
    calculateKmeans=False
    runLDA=True
    if visualWords:
        get_visual_words_and_save('Normal Crowds Testing')
    if calculateKmeans:
        kmeans=k_means(10)
        create_lda_model_from_visual_words(kmeans,'Test Dataset Crowd')
    if runLDA:
        latent_dirichlet_allocation(30)






import numpy as np
import math
from numba import njit, prange



def performance_roc(hist_genuine, hist_imposter, min_level=3, max_level=9):

    # thresholds 0 ~ 100000
    thresholds = np.arange(int(1e5), 0, -1)
    far_list = np.ones(len(thresholds)+1)
    tar_list = np.ones(len(thresholds)+1)
    cum_genuine = 0
    cum_imposter = 0       

    total_genuine = int(sum(hist_genuine))
    total_imposter = int(sum(hist_imposter))
    
    frr_far_diff = 1
    eer_threshold = 1e5
    roc_result = '\n'
    frr_far = [None] * (max_level-min_level+1)
    frr_far_th = [None] * (max_level-min_level+1)

    for threshold in thresholds:
        far = float(cum_imposter + hist_imposter[threshold])/total_imposter
        frr = float(total_genuine - cum_genuine)/total_genuine
        far_list[threshold] = far
        tar_list[threshold] = 1-frr
        
        for idx in range((max_level-min_level+1)):
            record_frr_far(frr, far, frr_far, frr_far_th, threshold, idx+min_level, min_level)
            
        if (abs(far - frr) < frr_far_diff):
            frr_far_diff = abs(far - frr)
            eer = (far + frr) / 2
            eer_threshold = threshold
        
        cum_genuine += hist_genuine[threshold]
        cum_imposter += hist_imposter[threshold]
    
    for idx in range((max_level-min_level+1)):
        roc_result += f"- FRR @ FAR{idx+min_level} {100 * frr_far[idx]:6.3f}%, (Threshold = {frr_far_th[idx] / 1e5:.5f})  \n"
    
    roc_result += "- EER {0:6.3f}%, (Threshold = {1:.5f})\n".format(100 * eer, eer_threshold / 1e5)
    roc_result += "- Total count = {:,}\n".format(total_genuine + total_imposter)
    roc_result += "- Total genuine count = {:,}\n".format(total_genuine)
    roc_result += "- Total imposter count = {:,}\n".format(total_imposter)
    
        
    return roc_result, eer_threshold


@njit(parallel=True)
def performance_acc(score_list, label_list, th):
    fr = 0
    fa = 0
    for i in prange(len(score_list)):
        score = score_list[i]
        label = label_list[i]
        
        if (score <= th / 1e5) and (label == 1): fr += 1
        if (score > th / 1e5) and (label == 0): fa += 1
        
    ACC = (1 - (fa + fr)/(len(score_list)))*100

    return ACC

@njit(parallel=True)
def pair_score(embedding_1, embedding_2, labels, metric='euclidean', min_level=3, max_level=9):

    num_total = len(labels)
    score_list = np.zeros(num_total)
    
    # ============================================
    # Calculate FAR and FRR
    # ============================================
    hist_genuine = np.zeros(100001)
    hist_imposter = np.zeros(100001)
    
    assert metric in ['euclidean', 'cosine'], 'Invalid metric !!!'
    if metric == 'euclidean':
        # ============================================
        # calculate cross matching scores and stack as a histogram
        # ============================================
        for i in prange(num_total):
            sum_diff = 0
            for k in prange(embedding_1.shape[1]):
                sum_diff += math.pow(embedding_1[i, k] - embedding_2[i, k], 2)
            score = (1 - sum_diff/4.)
            hist_idx = int((1e5-1.) * score)
            
            if labels[i]:
                hist_genuine[hist_idx] += 1
            else:
                hist_imposter[hist_idx] += 1
                
            score_list[i] = score
        
    return hist_genuine, hist_imposter, score_list


@njit(parallel=True)
def cross_score(embeddings, labels, metric='euclidean'):
    assert metric in ['euclidean', 'cosine'], 'Invalid metric !!!'    
    hist_genuine = np.zeros(100001)
    hist_imposter = np.zeros(100001)
    
    num_total = embeddings.shape[0]
    score_list = np.zeros(int((num_total-1)*num_total/2))
    label_list = np.zeros(int((num_total-1)*num_total/2))
    
    if metric == 'euclidean':  
        # ============================================
        # calculate cross matching scores and stack as a histogram
        # ============================================
        l = 0
        for i in prange(num_total):
            for j in prange(i):
                sum_diff = 0
                for k in prange(embeddings.shape[1]):
                    sum_diff += math.pow(embeddings[j, k] - embeddings[i, k], 2)
                score = (1 - sum_diff/4.)
                hist_idx = int((1e5-1.) * score)

                score_list[l] = score
                if labels[j] == labels[i]:
                    hist_genuine[hist_idx] += 1
                    label_list[l] = 1
                else:
                    hist_imposter[hist_idx] += 1
                    
                # print(score_list[k])
                # print(label_list[k])
                l += 1
                
    return hist_genuine, hist_imposter, score_list, label_list


def record_frr_far(frr, far, frr_far_list, th_list, th, security_level, min_level):
    if (far <= float(f'1e-{security_level}')):
        if (frr_far_list[security_level-min_level] is None) or frr < frr_far_list[security_level-min_level]:
            frr_far_list[security_level-min_level] = frr
            th_list[security_level-min_level] = th

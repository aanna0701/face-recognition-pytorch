import numpy as np
import math

def performance(embedding_1, embedding_2, label_list, metric='euclidean', min_level=3, max_level=9):

    num_total = len(label_list)
    
    # ============================================
    # Calculate FAR and FRR
    # ============================================
    hist_genuine = np.zeros(100001)
    hist_imposter = np.zeros(100001)
    cum_genuine = 0
    cum_imposter = 0
    
    # print(embedding_1.shape)
    # print(label_list.shape)
    
    assert metric in ['euclidean', 'cosine'], 'Invalid metric !!!'
    if metric == 'euclidean':
        sub = np.subtract(embedding_1, embedding_2)
        score_list = 1 - np.sum(np.square(sub), 1)/4.
        
    elif metric == 'cosine':
        score_list = (1 + np.sum(np.multiply(embedding_1, embedding_2), 1))/2 
        
    
    
    for i in range(num_total):
        label = label_list[i]
        score = math.floor(score_list[i] * 1e5)
        if label:
            hist_genuine[score] += 1
        else:
            hist_imposter[score] += 1
            
    

    # thresholds 0 ~ 100000
    thresholds = np.arange(int(1e5), 0, -1)
    far_list = np.ones(len(thresholds)+1)
    tar_list = np.ones(len(thresholds)+1)

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
        # if frr_far[idx] is None:
        #     frr_far[idx] = 1.
        #     frr_far_th[idx] = 1e5
        roc_result += f"- FRR @ FAR{idx+min_level} {100 * frr_far[idx]:6.3f}%, (Threshold = {frr_far_th[idx] / 1e5:.5f})  \n"
    
    roc_result += "- EER {0:6.3f}%, (Threshold = {1:.5f})\n".format(100 * eer, eer_threshold / 1e5)
    roc_result += "- Total count = {:,}\n".format(total_genuine + total_imposter)
    roc_result += "- Total genuine count = {:,}\n".format(total_genuine)
    roc_result += "- Total imposter count = {:,}\n".format(total_imposter)
    
    fr = 0
    fa = 0
    for i in range(num_total):
        score = score_list[i]
        label = label_list[i]
        
        if (score < eer_threshold / 1e5) and (label == 1): fr += 1
        if (score > eer_threshold / 1e5) and (label == 0): fa += 1
        
    ACC = (1 - (fa + fr)/(num_total-1))*100
        
    return roc_result, ACC


def record_frr_far(frr, far, frr_far_list, th_list, th, security_level, min_level):
    if (far <= float(f'1e-{security_level}')):
        if (frr_far_list[security_level-min_level] is None) or frr < frr_far_list[security_level-min_level]:
            frr_far_list[security_level-min_level] = frr
            th_list[security_level-min_level] = th

import numpy as np
import os
import os.path as p
from Utils import *

def f1_score(target,output, threshold):
    targetBinary = np.where(target.detach().cpu()==1,1,0)
    outputBinary = np.where(output.detach().cpu()>=threshold,1,0)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for actual,predicted in zip(targetBinary,outputBinary):
        if actual == 1 and predicted == 1:
            TP +=1
        elif actual == 0 and predicted == 0:
            TN+=1
        elif actual == 0 and predicted == 1:
            FP+=1
        elif actual == 1 and predicted == 0:
            FN+=1
    if TP == 0:
        precision = 0
        recall = 0
        f1 = 0
        accuracy = 0
    else:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f1 = (2*precision*recall)/(precision+recall)
        accuracy = (TP+TN)/(TP+TN+FP+FN)
    return f1,accuracy


def f1_score_test_analysis(target,output):
    targetBinary = target
    outputBinary = output
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for actual,predicted in zip(targetBinary,outputBinary):
        if actual == 1 and predicted == 1:
            TP +=1
           # print('Updated TP')
        elif actual == 0 and predicted == 0:
            TN+=1
        elif actual == 0 and predicted == 1:
            FP+=1
        elif actual == 1 and predicted == 0:
            FN+=1
    if TP == 0:
        precision = 0
        recall = 0
        f1 = 0
        accuracy = 0
    else:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f1 = (2*precision*recall)/(precision+recall)
        accuracy = (TP+TN)/(TP+TN+FP+FN)
    return f1,precision, recall, TP, FP, TN, FN

def get_score_reports_average(f1_scores, precisios, recalls, TPs, FPs, TNs, FNs, outputs_binary_all_class, outputDirectory):
    if p.exists(outputDirectory) == False:
        os.mkdir(outputDirectory)

    save_data(f1_scores, outputDirectory+'/fscore.pkl')
    save_data(precisios, outputDirectory + '/precisios.pkl')
    save_data(recalls, outputDirectory + '/recalls.pkl')
    save_data(TPs, outputDirectory + '/TPs.pkl')
    save_data(FPs, outputDirectory + '/FPs.pkl')
    save_data(TNs, outputDirectory + '/TNs.pkl')
    save_data(FNs, outputDirectory + '/FNs.pkl')
    save_data(outputs_binary_all_class, outputDirectory + '/outputs_binary_all_class.pkl')

    print(len(f1_scores))

    print('Precision', sum(precisios.values()) / len(precisios))
    print('Recall', sum(recalls.values()) / len(recalls))
    print('Macro f1', sum(f1_scores.values())/len(f1_scores))
    print('Micro f1', calculate_micro_f1_scores(TPs, FPs, FNs))

    with open(outputDirectory +'/results.txt', 'w') as file:
        file.write('Precision '+str(sum(precisios.values()) / len(precisios)) + '\n')
        file.write('Recall '+str(sum(recalls.values()) / len(recalls)) + '\n')
        file.write('Macro f1 '+str(sum(f1_scores.values())/len(f1_scores)) + '\n')
        file.write('Micro f1 '+str(calculate_micro_f1_scores(TPs, FPs, FNs)) + '\n')

def calculate_micro_f1_scores(TPs, FPs, FNs):
    all_TP = sum(TPs.values())
    all_FP = sum(FPs.values())
    all_FN = sum(FNs.values())
    all_precision = all_TP / (all_TP + all_FP)
    all_recall = all_TP / (all_TP + all_FN)
    micro_f1 = (2 * all_precision * all_recall) / (all_precision + all_recall)
    return micro_f1



def f1_score_for_flat(target,output, threshold):
    targetBinary = np.where(target.detach().cpu()==1,1,0)
    outputBinary = np.where(output.detach().cpu()>=threshold,1,0)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for actual,predicted in zip(targetBinary,outputBinary):
        if actual == 1 and predicted == 1:
            TP +=1
           # print('Updated TP')
        elif actual == 0 and predicted == 0:
            TN+=1
        elif actual == 0 and predicted == 1:
            FP+=1
        elif actual == 1 and predicted == 0:
            FN+=1
    if TP == 0:
        precision = 0
        recall = 0
        f1 = 0
        accuracy = 0
    else:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f1 = (2*precision*recall)/(precision+recall)
        accuracy = (TP+TN)/(TP+TN+FP+FN)
    return f1, precision, recall, TP, FP, TN, FN

def f1_score_all_classes(targets_all_classes, outputs_all_classes, thresholds, trainOrTest):
    targets_all_classes = targets_all_classes.transpose(0, 1)
    outputs_all_classes = outputs_all_classes.transpose(0, 1)
    scores = []
    best_thresholds = []
    TP = []
    FP = []
    TN = []
    FN = []
    precisions = []
    recalls = []
    if trainOrTest == 'train':
        for i in range(len(targets_all_classes)):
            cat_scores = []
            for thres in thresholds:
                score, _, _, _, _, _, _ = f1_score_for_flat(targets_all_classes[i], outputs_all_classes[i], thres)
                cat_scores.append(score)
            current_best_threshold_index = np.argmax(cat_scores)
            current_best_score = cat_scores[current_best_threshold_index]
            best_threshold = thresholds[current_best_threshold_index]
            scores.append(current_best_score)
            best_thresholds.append(best_threshold)
        return scores, best_thresholds
    else:
        for i in range(len(targets_all_classes)):
            score, precision_i, recall_i, TP_i, FP_i, TN_i, FN_i = f1_score_for_flat(targets_all_classes[i],
                                                                            outputs_all_classes[i], thresholds[i])
            scores.append(score)
            TP.append(TP_i)
            FP.append(FP_i)
            TN.append(TN_i)
            FN.append(FN_i)
            precisions.append(precision_i)
            recalls.append(recall_i)

        return scores, precisions, recalls, TP, FP, TN, FN

def calculate_micro_f1_scores_flat(TPs, FPs, FNs):
    all_TP = sum(TPs)
    all_FP = sum(FPs)
    all_FN = sum(FNs)
    all_precision = all_TP / (all_TP + all_FP)
    all_recall = all_TP / (all_TP + all_FN)
    micro_f1 = (2 * all_precision * all_recall) / (all_precision + all_recall)
    return micro_f1

def get_score_reports_average_flat(f1_scores, precisios, recalls, TPs, FPs, TNs, FNs, outputDirectory):
    if p.exists(outputDirectory) == False:
        os.mkdir(outputDirectory)

    save_data(f1_scores, outputDirectory+'/fscore.pkl')
    save_data(precisios, outputDirectory + '/precisios.pkl')
    save_data(recalls, outputDirectory + '/recalls.pkl')
    save_data(TPs, outputDirectory + '/TPs.pkl')
    save_data(FPs, outputDirectory + '/FPs.pkl')
    save_data(TNs, outputDirectory + '/TNs.pkl')
    save_data(FNs, outputDirectory + '/FNs.pkl')

    print(len(f1_scores))

    print('Precision', sum(precisios) / len(precisios))
    print('Recall', sum(recalls) / len(recalls))
    print('Macro f1', sum(f1_scores)/len(f1_scores))
    print('Micro f1', calculate_micro_f1_scores_flat(TPs, FPs, FNs))

    # with open(outputDirectory+'/results.txt', 'w') as file:
    #     file.write('Precision '+ str(sum(precisios) / len(precisios)))
    #     file.write('Recall ' + str(sum(recalls) / len(recalls)))
    #     file.write('Macro f1 ' + str(sum(f1_scores)/len(f1_scores)))
    #     file.write('Micro f1 ' + str(calculate_micro_f1_scores_flat(TPs, FPs, FNs)))

    with open(outputDirectory +'/results.txt', 'w') as file:
        file.write('Precision '+str(sum(precisios) / len(precisios)) + '\n')
        file.write('Recall '+str(sum(recalls) / len(recalls)) + '\n')
        file.write('Macro f1 '+str(sum(f1_scores)/len(f1_scores)) + '\n')
        file.write('Micro f1 '+str(calculate_micro_f1_scores_flat(TPs, FPs, FNs)) + '\n')



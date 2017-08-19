from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def get_classification_report(pred_targets, true_targets):
    int_pred_targets, int_true_targets = convert_to_same_format(pred_target, true_targets)
    return classification_report(int_true_targets, int_pred_targets)

def get_performance(pred_targets, true_targets, average='weighted'):
    int_pred_targets, int_true_targets = convert_to_same_format(pred_target, true_targets)
    return precision_recall_fscore_support(int_true_targets, int_pred_targets, average=average)

def get_accuracy(pred_targets, true_targets):
    int_pred_targets, int_true_targets = convert_to_same_format(pred_target, true_targets)
    # tot_correct = 0
    # for i in range(len(pred_targets)):
    #     if true_targets[i] == pred_targets[i]:
    #         # print("%d -> %d - %s -> %s" % (true_targets[i], pred_targets[i], type(true_targets[i]), type(pred_targets[i])))
    #         tot_correct += 1
    # test_accuracy = tot_correct/len(true_targets)
    return accuracy_score(int_true_targets, int_pred_targets)

def convert_to_same_format(pred_target, true_targets):
    int_pred_targets = []
    int_true_targets = []
    for i in range(len(pred_targets)):
        int_pred_targets.append(int(pred_targets[i]))
        int_true_targets.append(int(true_targets[i]))
    return int_pred_targets, int_true_targets


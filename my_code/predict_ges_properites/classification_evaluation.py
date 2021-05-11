from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch

def evaluation(prediction, truth):
    """
    Evaluate accuracy and f1 score for a given predictions

    Args:
        prediction:   predictions from the model (already rounded to 0 and 1_
        truth:        true labels for the same samples

    Returns:
        log:          log dictionary containing accuracy, precision, recall and f1 score

    """

    # find dimensionality of this feature
    feat_dim = truth.shape[1]

    # Get accuracy for a given feature
    acc_sum = 0
    f1_sum = 0
    prec_sum = 0
    recall_sum = 0

    n_present_features = feat_dim

    prefix = "feature_"

    log = {}

    print("\nValidating on ", truth.shape[0], " samples with ", torch.sum(truth.float()), " ones")

    for label in range(feat_dim):

        # ignore features which were basically not present in the validation set
        if torch.sum(truth[:, label]) < 5:
            print("\nIGNORING feature ", label, " in this validation as it was present only in ", sum(truth[:, label]),
                  " samples")
            n_present_features -= 1
            continue

        label_acc = accuracy_score(truth[:, label], prediction[:, label])
        log['Acc/' + prefix + str(label)] = label_acc
        acc_sum += label_acc

        # f1 score is even more important
        label_f1 = f1_score(truth[:, label], prediction[:, label])
        log['F1/' + prefix + str(label)] = label_f1
        f1_sum += label_f1

        # check also precision
        label_prec = precision_score(truth[:, label], prediction[:, label])
        log['Prec/' + prefix + str(label)] = label_prec
        prec_sum += label_prec

        # check also recall
        label_recall = recall_score(truth[:, label], prediction[:, label])
        log['Rec/' + prefix + str(label)] = label_recall
        recall_sum += label_recall

    if n_present_features == 0:
        print("Will skip this fold because there are no properties to evaluate on")
        return log

    mean_acc_phr = acc_sum / (n_present_features )
    log["Acc/" + prefix + "av"] = mean_acc_phr

    mean_f1_phr = f1_sum / (n_present_features)
    log["F1/" + prefix + "av"] = mean_f1_phr

    log["Prec/" + prefix + "av"] =  prec_sum / (n_present_features)

    log["Rec/" + prefix + "av"] = recall_sum / (n_present_features)
    
    return log

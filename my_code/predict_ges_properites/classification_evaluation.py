from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluation(prediction, truth, feat_dim):
    """
    Evaluate accuracy and f1 score for a given predictions

    Args:
        prediction:   predictions from the model (already rounded to 0 and 1_
        truth:        true labels for the same samples
        feat_dim:     dimensionality of this feature

    Returns:
        log:          log dictionary containing accuracy, precision, recall and f1 score

    """

    # Get accuracy for a given feature
    acc_sum = 0
    f1_sum = 0
    prec_sum = 0
    recall_sum = 0

    prefix = "feature_"

    log = {}

    for label in range(feat_dim):
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

    mean_acc_phr = acc_sum / (feat_dim )
    log["Acc/" + prefix + "_av"] = mean_acc_phr

    mean_f1_phr = f1_sum / (feat_dim)
    log["F1/" + prefix + "_av"] = mean_f1_phr

    log["Prec/" + prefix + "_av"] =  prec_sum / (feat_dim)

    log["Rec/" + prefix + "_av"] = recall_sum / (feat_dim)

    return log
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_phrase(prediction, truth):

    # Get accuracy for phrase
    acc_sum = 0
    f1_sum = 0
    prec_sum = 0
    recall_sum = 0

    phrase_length = 7

    prefix = "phrase_"

    log = {}

    for label in range(phrase_length):
        if label == 2 or label == 6 or label == 3:
            continue
        label_acc = accuracy_score(truth[:, label], prediction[:, label])
        log['Acc/' + prefix + str(label)] = label_acc
        acc_sum += label_acc

        # f1 score is even more important
        label_f1 = f1_score(truth[:, label], prediction[:, label])
        log['F1/' + prefix + str(label)] =  label_f1
        f1_sum += label_f1

        # check also precision
        label_prec = precision_score(truth[:, label], prediction[:, label])
        log['Prec/' + prefix + str(label)] = label_prec
        prec_sum += label_prec

        # check also recall
        label_recall = recall_score(truth[:, label], prediction[:, label])
        log['Rec/' + prefix + str(label)] = label_recall
        recall_sum += label_recall

    mean_acc_phr = acc_sum / (phrase_length - 3)
    log["Acc/" + prefix + "_av"] = mean_acc_phr

    mean_f1_phr = f1_sum / (phrase_length - 3)
    log["F1/" + prefix + "_av"] = mean_f1_phr

    log["Prec/" + prefix + "_av"] =  prec_sum / (phrase_length - 3)

    log["Rec/" + prefix + "_av"] = recall_sum / (phrase_length - 3)

    return log


def evaluate_g_semantic(prediction, truth):

    # Get accuracy for phrase
    acc_sum = 0
    f1_sum = 0
    prec_sum = 0
    recall_sum = 0

    semant_length = 5

    prefix = "semantic_"

    log = {}

    for label in range(semant_length):
        if label == 1:
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

    mean_acc_phr = acc_sum / (semant_length - 1)
    log["Acc/" + prefix + "_av"] = mean_acc_phr

    mean_f1_phr = f1_sum / (semant_length - 1)
    log["F1/" + prefix + "_av"] = mean_f1_phr

    log["Prec/" + prefix + "_av"] =  prec_sum / (semant_length - 1)

    log["Rec/" + prefix + "_av"] = recall_sum / (semant_length - 1)

    return log


def evaluate_s_semantic(prediction, truth):

    # Get accuracy for s semantic
    acc_sum = 0
    f1_sum = 0
    prec_sum = 0
    recall_sum = 0

    semant_length = 6

    prefix = "sp_semantic_"

    log = {}

    for label in range(semant_length):
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

    mean_acc_phr = acc_sum / semant_length
    log["Acc/" + prefix + "_av"] = mean_acc_phr

    mean_f1_phr = f1_sum / semant_length
    log["F1/" + prefix + "_av"] = mean_f1_phr

    log["Prec/" + prefix + "_av"] = prec_sum / semant_length

    log["Rec/" + prefix + "_av"] = recall_sum / semant_length

    return log


def evaluate_practice(prediction, truth):

    # Get accuracy for practice
    acc_pract_sum = 0
    f1_pract_sum = 0
    prec_sum = 0
    recall_sum = 0
    phrase_length = 7
    prefix = "practice_"

    log = {}

    for label in range(phrase_length, prediction.shape[1]):
        if label == 10 or label == 14 or label == 18:
            continue
        label_acc = accuracy_score(truth[:, label], prediction[:, label])
        log['Acc/' + prefix + str(label)] = label_acc
        acc_pract_sum += label_acc

        # f1 score is even more important
        label_f1 = f1_score(truth[:, label], prediction[:, label])
        log['F1/' + prefix + str(label)] = label_f1
        f1_pract_sum += label_f1

        # check also precision
        label_prec = precision_score(truth[:, label], prediction[:, label])
        log['Prec/' + prefix + str(label)] = label_prec
        prec_sum += label_prec

        # check also recall
        label_recall = recall_score(truth[:, label], prediction[:, label])
        log['Rec/' + prefix + str(label)] = label_recall
        recall_sum += label_recall
    
    mean_pract_acc = acc_pract_sum / (prediction.shape[1] - phrase_length - 3)
    log["Acc/" + prefix + "_av"] = mean_pract_acc

    mean_f1_pract = f1_pract_sum / (prediction.shape[1] - phrase_length - 3)
    log["F1/" + prefix + "_av"] = mean_f1_pract

    log["Prec/" + prefix + "_av"] =  prec_sum / (prediction.shape[1] - phrase_length - 3)
    log["Rec/" + prefix + "_av"] = recall_sum / (prediction.shape[1] - phrase_length - 3)
    
    return log


def evaluate_phase(prediction, truth):
    # Get accuracy for phase
    acc_sum = 0
    f1_sum = 0
    prec_sum = 0
    recall_sum = 0
    phrase_length = 5
    prefix = "phase_"

    log = {}

    for label in range(phrase_length):
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

    mean_pract_acc = acc_sum / phrase_length
    log["Acc/" + prefix + "_av"] = mean_pract_acc

    mean_f1_pract = f1_sum / phrase_length
    log["F1/" + prefix + "_av"] = mean_f1_pract

    log["Prec/" + prefix + "_av"] = prec_sum / phrase_length
    log["Rec/" + prefix + "_av"] = recall_sum / phrase_length

    return log
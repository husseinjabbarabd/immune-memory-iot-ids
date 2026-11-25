from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score


def report(yt, yp):
    return dict(acc=accuracy_score(yt, yp), f1w=f1_score(yt, yp, average='weighted'), f1m=f1_score(yt, yp, average='macro'),
                pw=precision_score(yt, yp, average='weighted', zero_division=0), rw=recall_score(yt, yp, average='weighted'),
                pm=precision_score(yt, yp, average='macro', zero_division=0), rm=recall_score(yt, yp, average='macro'),
                kappa=cohen_kappa_score(yt, yp), recall=recall_score(yt, yp, average=None).tolist(),
                f1=f1_score(yt, yp, average=None).tolist())

import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import roc_curve, auc

timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S").replace("'", "")


def create_loss_image(loss, epoch):
    '''
    生成loss曲线图
    :param loss:loss数组
    :param epoch: epoch数组
    :return:
    '''
    plt.figure(figsize=(10, 5))
    plt.plot(epoch, loss, 'r-', label=u'GCN')
    plt.title('loss vs epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('./figs/loss_' + timestamp + '.png')
    plt.show()
    plt.close()


def create_acc_image(acc, epoch):
    '''
    生成acc曲线图
    :param acc: acc数组
    :param epoch: epoch数组
    :return:
    '''
    plt.figure(figsize=(10, 5))
    plt.plot(epoch, acc, 'r-', label=u'GCN')
    plt.title('acc vs epoch')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.grid(True)
    plt.legend()
    plt.savefig('./figs/acc_' + timestamp + '.png')
    plt.show()
    plt.close()


def create_roc_image(y_label, y_pred):
    fpr, tpr, thersholds = roc_curve(y_label, y_pred, pos_label=1)
    for i, value in enumerate(thersholds):
        print("%f %f %f" % (fpr[i], tpr[i], value))

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

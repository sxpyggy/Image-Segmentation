# json文件指标含义
# {
#     "data_time": 0.0008994499999630534,
#      "eta_seconds": 1.1653430999999728,
#      "fast_rcnn/cls_accuracy": 0.8815104166666666,  # 分类准确率
#      "fast_rcnn/false_negative": 0.5488476218438051,  # 假阴率
#      "fast_rcnn/fg_cls_accuracy": 0.4511523781561949,  # foreground分类准确率
#      "iteration": 499,
#      "loss_box_reg": 0.5407436788082123,  # box回归的损失
#      "loss_cls": 0.2957667261362076,  # 分类损失
#      "loss_rpn_cls": 0.02278049197047949,
#      "loss_rpn_loc": 0.01768689788877964,
#      "lr": 0.00999002,
#      "roi_head/num_bg_samples": 215.33333333333334,
#      "roi_head/num_fg_samples": 40.666666666666664,
#      "rpn/num_neg_anchors": 249.16666666666669,
#      "rpn/num_pos_anchors": 6.833333333333334,
#      "time": 1.135507050000058,
#      "total_loss": 0.9146622531116009  # 总损失
#  }


import json
import matplotlib.pyplot as plt
import datetime
import pandas as pd

pwd = './output0731_2/'
log_file = pwd + 'metrics.json'


def analyse_log(log_file):
    iters = []
    cls_acc = []  # 使用fast_rcnn/cls_accuracy  分类准确率
    fnr = []  # 使用fast_rcnn/false_negative
    lr = []  # 学习率
    total_loss = []

    with open(log_file) as f:
        for line in f:
            tmp = json.loads(line)
            iters.append(tmp["iteration"])
            cls_acc.append(tmp["fast_rcnn/cls_accuracy"])
            fnr.append(tmp["fast_rcnn/false_negative"])
            lr.append(tmp["lr"])
            total_loss.append(tmp["total_loss"])

    df = {"iters": iters, "cls_accuracy": cls_acc,
          "False Negative Ratio": fnr, "learning rate": lr, "total_loss": total_loss}
    df = pd.DataFrame(df)

    # 绘图部分
    fig = plt.figure(figsize=(8, 6))
    ax2 = fig.add_subplot(111)
    ax1 = ax2.twinx()
    # acc
    ax1.plot(iters, cls_acc, color='red', label='cls_acc')
    # fnr
    ax1.plot(iters, fnr, color='orange', label='fnr')

    # loss
    ax1.plot(iters, total_loss, color='olive', label='total_loss')

    # 设置坐标轴范围
    ax1.set_xlim((0, iters[-1]))
    ax1.set_ylim((0, 1.2))

    # 设置坐标轴、图片名称
    ax1.set_xlabel('iterations')
    today = datetime.date.today().strftime('%y%m%d')
    ax1.set_title(today)
    ax1.legend(loc='upper right')

    # lr
    ax2.plot(iters, lr, color='black', label='lr')
    ax2.set_ylim([0, max(lr) * 1.1])
    ax2.legend(loc='center right')

    ax1.set_ylabel('loss and accuracy')
    ax2.set_ylabel('learning rate')

    plt.savefig(pwd + today + '.png')
    plt.show()
    return df


if __name__ == '__main__':
    log_df = analyse_log(log_file)
    log_df.to_csv(pwd + '/log.csv')  # 关键要记录参数信息

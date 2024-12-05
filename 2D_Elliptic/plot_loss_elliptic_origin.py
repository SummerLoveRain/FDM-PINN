# 读取xlsx文件内容，得到数据
import numpy as np
from plot.line import plot_line


# step值越过多少个点再记录
def read_PINN_log(PINN_log_path, step=1):
    # INFO:root:Adam Iter 10 Loss 432.10754 equation_loss 429.20953 u_loss 0.43641347 lambda_loss 2.4519458 drive_loss 0.009671012 LR 0.008
    total_loss = []
    loss_region = []
    loss_Gamma = []
    loss_gamma = []
    with open(PINN_log_path, 'r') as fs:
        line_num = 0
        while True:
            line = fs.readline()  # 整行读取数据
            if not line:
                break
            # 读取loss
            if 'Iter ' in line:
                line_num = line_num + 1
                if line_num % step != 0:
                    continue
                # 总误差
                loss = line.split(' Loss ')[-1].split(' loss_region ')[0]
                if loss is not None:
                    total_loss.append(float(loss))

                loss = line.split(' loss_region ')[-1].split(
                    ' loss_Gamma ')[0]
                if loss is not None:
                    loss_region.append(float(loss))

                loss = line.split(' loss_Gamma ')[-1].split(
                    ' loss_gamma ')[0]
                if loss is not None:
                    loss_Gamma.append(float(loss))

                loss = line.split(
                    ' loss_gamma ')[-1].split(' LR ')[0]
                if loss is not None:
                    loss_gamma.append(float(loss))

    # all_loss = [total_loss, loss_region, loss_Gamma, loss_gamma]
    all_loss = [total_loss, loss_region, loss_Gamma, loss_gamma]
    return all_loss


if __name__ == "__main__":
    root_path = 'c:/softwarefiles/VSCodeProjects/FictitiousDomain/2D_Elliptic/data/'
    datas = []
    TASK_NAME = 'task_pinn_origin'
    TIME_STR = '20220504_090922'

    PINN_log_path = root_path + '/' + TASK_NAME + '/' + TIME_STR + '/log.txt'

    all_loss = read_PINN_log(
        PINN_log_path=PINN_log_path, step=1)

    Num_Epoch = len(all_loss[0])
    epochs = [i for i in range(1, Num_Epoch + 1)]

    for loss in all_loss:
        data = np.stack((epochs, loss), 1)
        datas.append(data)

    data_labels = ['total', 'region', 'Gamma', 'gamma']
    # data_labels = None

    xy_labels = ['Epoch', 'Loss']
    plot_line(datas=datas,
              data_labels=data_labels,
              xy_labels=xy_labels,
              title=None,
              file_name=root_path + '/' + TASK_NAME + '/' + TIME_STR + '/loss',
              ylog=True)
    print('done')

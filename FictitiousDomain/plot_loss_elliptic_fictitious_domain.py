# 读取xlsx文件内容，得到数据
import numpy as np
from plot.line import plot_line


# step值越过多少个点再记录
def read_PINN_log(PINN_log_path, step=1):
    # INFO:root:Adam Iter 10 Loss 432.10754 equation_loss 429.20953 u_loss 0.43641347 lambda_loss 2.4519458 drive_loss 0.009671012 LR 0.008
    total_loss = []
    loss_Omega = []
    loss_omega1 = []
    loss_omega2 = []
    loss_Gamma = []
    loss_gamma = []
    loss_J = []
    with open(PINN_log_path, 'r') as fs:
        line_num = 0
        while True:
            line = fs.readline()  # 整行读取数据
            if not line:
                break
            # INFO:root:Adam Iter 10 Loss 26513.96 loss_Omega 4752.909 loss_omega1 21756.22 loss_omega2 1.4635822 loss_J 2.202965 loss_Gamma 0.0 loss_gamma 1.166757 LR 0.001
            # 读取loss
            if 'Iter ' in line:
                line_num = line_num + 1
                if line_num % step != 0:
                    continue
                # 总误差
                loss = line.split(' Loss ')[-1].split(' loss_Omega ')[0]
                if loss is not None:
                    total_loss.append(float(loss))

                loss = line.split(' loss_Omega ')[-1].split(
                    ' loss_omega1 ')[0]
                if loss is not None:
                    loss_Omega.append(float(loss))

                loss = line.split(' loss_omega1 ')[-1].split(
                    ' loss_omega2 ')[0]
                if loss is not None:
                    loss_omega1.append(float(loss))

                loss = line.split(' loss_omega2 ')[-1].split(
                    ' loss_J ')[0]
                if loss is not None:
                    loss_omega2.append(float(loss))

                loss = line.split(' loss_J ')[-1].split(
                    ' loss_Gamma ')[0]
                if loss is not None:
                    loss_J.append(float(loss))

                loss = line.split(' loss_Gamma ')[-1].split(
                    ' loss_gamma ')[0]
                if loss is not None:
                    loss_Gamma.append(float(loss))

                loss = line.split(
                    ' loss_gamma ')[-1].split(' LR ')[0]
                if loss is not None:
                    loss_gamma.append(float(loss))

    # all_loss = [total_loss, loss_region, loss_Gamma, loss_gamma]
    all_loss = [total_loss, loss_Omega, loss_omega1, loss_omega2, loss_gamma]
    return all_loss


if __name__ == "__main__":
    root_path = 'c:/softwarefiles/VSCodeProjects/FictitiousDomain/data/'
    datas = []
    TASK_NAME = 'task_pinn_fictitious_domain'
    TIME_STR = '20220504_103332'

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

    xy_labels = ['Epoch/10', 'Loss']
    plot_line(datas=datas,
              data_labels=data_labels,
              xy_labels=xy_labels,
              title=None,
              file_name=root_path + '/' + TASK_NAME + '/' + TIME_STR + '/loss',
              ylog=True)
    print('done')

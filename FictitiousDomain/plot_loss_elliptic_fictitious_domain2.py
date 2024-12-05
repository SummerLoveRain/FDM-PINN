# 读取xlsx文件内容，得到数据
import numpy as np
from plot.line import plot_line


# step值越过多少个点再记录
def read_PINN_log(PINN_log_path, step=1):
    loss_psi1 = []
    loss_psi2 = []
    loss_J = []
    loss_gamma = []
    with open(PINN_log_path, 'r') as fs:
        line_num = 0
        while True:
            line = fs.readline()  # 整行读取数据
            if not line:
                break
            # 读取loss
            if 'Iter ' in line:
                # INFO:root:Adam Iter 30 training_type 2 Loss 9145.063 loss_psi1 0.0 loss_psi2 0.0 loss_J 9145.063 loss_Gamma 0.0 loss_gamma 0.0 LR 0.001
                line_num = line_num + 1
                if line_num % step != 0:
                    continue
                # 总误差
                training_type = line.split(
                    ' training_type ')[-1].split(' Loss ')[0]
                if training_type == '0':

                    loss = line.split(' loss_psi1 ')[-1].split(
                        ' loss_psi2 ')[0]
                    if loss is not None:
                        loss_psi1.append(float(loss))
                if training_type == '1':
                    loss = line.split(' loss_psi2 ')[-1].split(
                        ' loss_J ')[0]
                    if loss is not None:
                        loss_psi2.append(float(loss))
                    loss = line.split(' loss_gamma ')[-1].split(
                        ' LR ')[0]
                    if loss is not None:
                        loss_gamma.append(float(loss))
                if training_type == '2':
                    loss = line.split(' loss_J ')[-1].split(
                        ' loss_Gamma ')[0]
                    if loss is not None:
                        loss_J.append(float(loss))

    # all_loss = [total_loss, loss_region, loss_Gamma, loss_gamma]
    all_loss = [loss_psi1, loss_psi2, loss_J, loss_gamma]
    return all_loss


if __name__ == "__main__":
    root_path = 'c:/softwarefiles/VSCodeProjects/FictitiousDomain/data/'
    datas = []
    TIME_STR = '20220504_124502'
    TASK_NAME = 'task_pinn_fictitious_domain2'

    PINN_log_path = root_path + '/' + TASK_NAME + '/' + TIME_STR + '/log.txt'

    all_loss = read_PINN_log(
        PINN_log_path=PINN_log_path, step=1)

    Num_Epoch = len(all_loss[0])
    epochs = [i for i in range(1, Num_Epoch + 1)]

    for loss in all_loss:
        data = np.stack((epochs, loss), 1)
        datas.append(data)

    data_labels = ['loss_psi1', 'loss_psi2', 'loss_J', 'loss_gamma']
    # data_labels = None

    xy_labels = ['Epoch', 'Loss']
    plot_line(datas=datas,
              data_labels=data_labels,
              xy_labels=xy_labels,
              title=None,
              file_name=root_path + '/' + TASK_NAME + '/' + TIME_STR + '/loss',
              ylog=True)
    print('done')

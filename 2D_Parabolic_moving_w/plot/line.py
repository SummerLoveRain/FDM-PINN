# pip install SciencePlots
import matplotlib.pyplot as plt

# plt.style.use(['science', 'ieee', 'grid'])
plt.style.use(['science', 'high-vis', 'grid'])


# datas:数据数组[[x, y]]
# data_labes: 每一组数据的标签
# xy_labels: xy轴标签
# title: 图片的标题, 为None则不画
# file_name: 文件保存名字, 默认保存到当前路径下面
# log: 是否以log的方式画图
def plot_line(datas,
              data_labels,
              xy_labels,
              title,
              file_name,
              xlog=False,
              ylog=False):
    fig, ax = plt.subplots()
    if data_labels is not None:
        for data, data_label in zip(datas, data_labels):
            x = data[:, 0]
            y = data[:, 1]
            ax.plot(x, y, label=data_label)
        ax.legend()
    else:
        for data in datas:
            x = data[:, 0]
            y = data[:, 1]
            ax.plot(x, y)
    ax.set(xlabel=xy_labels[0])
    ax.set(ylabel=xy_labels[1])
    ax.autoscale(tight=True)

    # xy轴是否以log的方式画图
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')

    # 设置标题
    if title is not None:
        ax.set_title(title)

    fig.savefig(file_name + '.png', dpi=300)

    # plt.show()

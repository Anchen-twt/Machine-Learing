import torch
from torch import nn
import d2l

# 定义真实的权重 true_w 和偏置 true_b
true_w = torch.tensor([2, -3.4])
true_b = 4.2

# 生成合成数据集 features 和 labels
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 定义批量大小
batch_size = 10

# 将数据集划分为大小为 batch_size 的小批量数据，并创建数据迭代器 data_iter
data_iter = d2l.load_array((features, labels), batch_size)

# 定义线性回归模型
net = nn.Sequential(nn.Linear(2, 1))  # 输入特征数为 2，输出特征数为 1

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)  # 权重初始化为均值为 0，标准差为 0.01 的正态分布
net[0].bias.data.fill_(0)  # 偏置初始化为 0

# 定义均方误差损失函数
loss = nn.MSELoss()

# 定义优化器，使用随机梯度下降 (SGD) 来更新模型参数，学习率为 0.03
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练次数
num_epochs = 3

# 开始训练循环
for epoch in range(num_epochs):
    for X, y in data_iter:
        # 前向传播计算预测值并计算损失
        l = loss(net(X), y)
        
        # 梯度清零，防止梯度累积
        trainer.zero_grad()
        
        # 反向传播计算梯度
        l.backward()
        
        # 使用优化器更新模型参数
        trainer.step()
        
    # 计算每个 epoch 结束后的整体损失
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

# 获取训练后的模型参数
w = net[0].weight.data
b = net[0].bias.data

# 输出参数估计误差：真实值 true_w 和 true_b 与估计值 w 和 b 之间的差异
print('w的估计误差：', true_w - w.reshape(true_w.shape))
print('b的估计误差：', true_b - b)
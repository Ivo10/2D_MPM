def compute_accuracy(output, target):
    """计算准确率"""
    predicted = (output >= 0.5).float()  # 将概率值转为二分类标签
    correct = (predicted == target).sum().item()
    accuracy = correct / target.size(0)
    return accuracy
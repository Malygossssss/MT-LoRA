def calculate_stats(a, b, c, d):
    """
    对输入的四个浮点数进行归一化计算，然后计算平均值和方差
    
    参数:
        a, b, c, d: 四个浮点数
    
    返回:
        mean: 平均值
        variance: 方差
    """
    # 对四个数分别进行归一化计算
    a = (a - 67.21) / 67.21
    b = (b - 61.93) / 61.93
    c = (c - 62.35) / 62.35
    d = (17.97 - d) / 17.97
    
    # 计算平均值
    mean = (a + b + c + d) / 4
    print(f"平均值: {mean}")
    
    # 计算方差
    variance = ((a - mean)**2 + (b - mean)**2 + (c - mean)**2 + (d - mean)**2) / 4
    print(f"方差: {variance}")
    
    return mean, variance


# 示例使用
if __name__ == "__main__":
    # # 输入四个浮点数
    # a = float(input("请输入a: "))
    # b = float(input("请输入b: "))
    # c = float(input("请输入c: "))
    # d = float(input("请输入d: "))

    # 方式1: 直接设置测试值（取消下面的注释使用）
    a, b, c, d = 70.82, 60.45, 65.88, 16.48
    
    # 调用函数
    mean, variance = calculate_stats(a, b, c, d)
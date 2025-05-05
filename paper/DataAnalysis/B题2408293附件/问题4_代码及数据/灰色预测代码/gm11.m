 % 函数作用：使用传统的GM(1,1)模型对数据进行预测
function [result, x0_hat, relative_residuals, eta] = gm11(x0, predict_num)

    n = length(x0); % 数据的长度
    x1 = cumsum(x0); % 计算一次累加值
    z1 = (x1(1:end-1) + x1(2:end)) / 2;  % 计算紧邻均值生成数列（长度为n-1）
    
    y = x0(2:end); 
    x = z1;
    
    k = ((n-1) * sum(x .* y) - sum(x) * sum(y)) / ((n-1) * sum(x .* x) - sum(x) * sum(x));
    b = (sum(x .* x) * sum(y) - sum(x) * sum(x .* y)) / ((n-1) * sum(x .* x) - sum(x) * sum(x));
    a = -k;
    
    % 注意：-a就是发展系数, b就是灰作用量
    disp('现在进行GM(1,1)预测的原始数据是: ')
    disp(mat2str(x0'))  % mat2str可以将矩阵或者向量转换为字符串显示
    disp(strcat('最小二乘法拟合得到的发展系数为', num2str(-a), '，灰作用量是', num2str(b)))
    disp('————————————————分割线——————————————————')
    
    x0_hat = zeros(n, 1);  
    x0_hat(1) = x0(1);   % x0_hat向量用来存储对x0序列的拟合值，这里先进行初始化
    for m = 1: n-1
        x0_hat(m+1) = (1 - exp(a)) * (x0(1) - b / a) * exp(-a * m);
    end
    
    result = zeros(predict_num, 1);  % 初始化用来保存预测值的向量
    for i = 1: predict_num
        result(i) = (1 - exp(a)) * (x0(1) - b / a) * exp(-a * (n + i - 1)); % 带入公式直接计算
    end
    
    % 计算绝对残差和相对残差
    absolute_residuals = x0(2:end) - x0_hat(2:end);  %从第二项开始计算残差
    relative_residuals = abs(absolute_residuals) ./ x0(2:end);  % 注意分子要加绝对值，而且要使用点除
    
    % 计算级比和级比偏差
    class_ratio = x0(2:end) ./ x0(1:end-1);  % 计算级比
    eta = abs(1 - (1 - 0.5 * a) / (1 + 0.5 * a) * (1 ./ class_ratio));  % 计算级比偏差
end
% 函数作用：使用新陈代谢的GM(1,1)模型对数据进行预测
function [result] = metabolism_gm11(x0, predict_num)

    result = zeros(predict_num, 1);  % 初始化用来保存预测值的向量
    for i = 1 : predict_num
        result(i) = gm11(x0, 1);  % 将预测一期的结果保存到result中
        x0 = [x0(2:end); result(i)];  % 更新x0向量
    end
end
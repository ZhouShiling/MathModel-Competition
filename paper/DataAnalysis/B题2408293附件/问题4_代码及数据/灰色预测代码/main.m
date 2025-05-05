clear;clc  % 清除命令窗口中的所有变量和清除命令窗口的内容。
% 年份序列和数据的初始值

data = [3945 
3937 
3985 
3894 
3907 
3932 
7259 
6614 
6615 
6608 
6705 
5279 
10296 
9618 
9086 
8087 
8175 
7465 
6739 
14244 
16000 
17412 
18320 
18663 
18873 
18798 
18825 
18724 
18450 
18376 
18158 
16970 
15040 
16427 
]';  % 赋初始数据
day = (1:1:length(data))';  % 横坐标表示年份序列，已知数据是1-7天，预测第8天
% 画出原始数据的时间序列图
figure(1);  %对画布进行编号，以为预测产生的画布不止一幅
plot(day,data,'o-'); 
grid on;  
set(gca,'xtick',day(1:1:end))  
xlabel('年份序列');  ylabel('医疗卫生机构数');  % 给坐标轴加上标签

% 进行数据的合法性检验。
ERROR = 0;
if sum(data<0) > 0  % data<0返回一个逻辑数组(0-1组成)，如果有数据小于0，则所在位置为1，如果原始数据均为非负数，那么这个逻辑数组中全为0，求和后也是0
    disp('灰色预测的时间序列中不能使用负数')
    ERROR = 1;
end

n = length(data);  % 计算原始数据的长度
disp(strcat('原始数据的长度为',num2str(n)))
if n<=3
    disp('数据量太小，无法进行灰色预测')
    ERROR = 1;
end

if n>10
    disp('数据量太多，灰色预测无法给出精准的预测')
end

if size(data,1) == 1
    data = data';
end
if size(day,1) == 1
    day = day';
end

% 进行准指数规律检验
if ERROR == 0   
    disp('————————————————分割线——————————————————')
    disp('准指数规律检验')
    x1 = cumsum(data);   
    rho = data(2:end) ./ x1(1:end-1);  
   
    figure(2)
    plot(day(2:end),rho,'o-',[day(2),day(end)],[0.5,0.5],'-'); grid on;
    text(day(end-1)+0.2,0.55,'临界线')   
    set(gca,'xtick',day(2:1:end))  
    xlabel('年份序列');  ylabel('原始数据的光滑度');  % 计算光滑比
    
    disp(strcat('指标1：光滑比小于0.5的数据占比为',num2str(100*sum(rho<0.5)/(n-1)),'%'))
    disp(strcat('指标2：除去前两个时期外，光滑比小于0.5的数据占比为',num2str(100*sum(rho(3:end)<0.5)/(n-3)),'%'))
    disp('参考标准：指标1一般要大于60%, 指标2要大于90%，你认为本例数据可以通过检验吗？')
    Judge = input('你认为可以通过准指数规律的检验吗？可以通过请输入1，不能请输入0：');
    if Judge == 0
        disp('灰色预测模型不适合此数据')
        ERROR = 1;
    end
    disp('————————————————分割线——————————————————')
end

% 根据数据的期数，将数据分为训练组和试验组。 
if ERROR == 0   
    if  n > 4  
        disp('因为原数据的期数大于4，所以我们可以将数据组分为训练组和试验组')   
        if n > 7
            test_num = 3;
        else
            test_num = 2;
        end
        train_data = data(1:end-test_num);  
        disp('训练数据是: ')
        disp(mat2str(train_data'))  
        test_data =  data(end-test_num+1:end); 
        disp('试验数据是: ')
        disp(mat2str(test_data'))  
        disp('————————————————分割线——————————————————')
        
        % 使用传统的GM(1,1)模型
        disp(' ')
        disp('传统的GM(1,1)模型预测的详细过程：')
        result1 = gm11(train_data, test_num); 
        disp(' ')
        % 新信息的GM(1,1)模型
        disp('下面是进行新信息的GM(1,1)模型预测的详细过程：')
        result2 = new_gm11(train_data, test_num); 
        disp(' ')
        % 新陈代谢的GM(1,1)模型
        disp('下面是进行新陈代谢的GM(1,1)模型预测的详细过程：')
        result3 = metabolism_gm11(train_data, test_num); 
        
        disp(' ')
        disp('————————————————分割线——————————————————')
        
        test_day = day(end-test_num+1:end);  
        figure(3)
        plot(test_day,test_data,'o-',test_day,result1,'*-',test_day,result2,'+-',test_day,result3,'x-'); grid on;
        set(gca,'xtick',day(end-test_num+1): 1 :day(end))  
        legend('试验组的真实数据','传统GM(1,1)预测结果','新信息GM(1,1)预测结果','新陈代谢GM(1,1)预测结果')  
        xlabel('年份序列');  ylabel('医疗卫生机构数');  
        
        SSE_1 = sum((test_data-result1).^2);
        SSE_2 = sum((test_data-result2).^2);
        SSE_3 = sum((test_data-result3).^2);
        disp(strcat('传统GM(1,1)对于试验组预测的误差平方和为',num2str(SSE_1)))
        disp(strcat('新信息GM(1,1)对于试验组预测的误差平方和为',num2str(SSE_2)))
        disp(strcat('新陈代谢GM(1,1)对于试验组预测的误差平方和为',num2str(SSE_3)))
        
        if SSE_1<SSE_2
            if SSE_1<SSE_3
                choose = 1;  
            else
                choose = 3;  
            end
        elseif SSE_2<SSE_3
            choose = 2;  
        else
            choose = 3;  
        end
        
        Model = {'传统GM(1,1)模型','新信息GM(1,1)模型','新陈代谢GM(1,1)模型'};
        disp(strcat('因为',Model(choose),'的误差平方和最小，所以我们应该选择其进行预测'))
        disp('————————————————分割线——————————————————')
        
        predict_num = input('请输入你要往后面预测的期数： ');

        [result, data_hat, relative_residuals, eta] = gm11(data, predict_num);  
        
        if choose == 2
            result = new_gm11(data, predict_num);
        end
        if choose == 3
            result = metabolism_gm11(data, predict_num);
        end
        
        disp('————————————————分割线——————————————————')
        disp('对原始数据的拟合结果：')
        for i = 1:n
            disp(strcat(num2str(day(i)), ' ： ',num2str(data_hat(i))))
        end
        disp(strcat('往后预测',num2str(predict_num),'期的得到的结果：'))
        for i = 1:predict_num
            disp(strcat(num2str(day(end)+i), ' ： ',num2str(result(i))))
        end
        
    else
        disp('因为数据只有4期，因此我们直接将三种方法的结果求平均即可~')
        predict_num = input('请输入你要往后面预测的期数： ');
        disp(' ')
        disp('***下面是传统的GM(1,1)模型预测的详细过程***')
        [result1, data_hat, relative_residuals, eta] = gm11(data, predict_num);
        disp(' ')
        disp('***下面是进行新信息的GM(1,1)模型预测的详细过程***')
        result2 = new_gm11(data, predict_num);
        disp(' ')
        disp('***下面是进行新陈代谢的GM(1,1)模型预测的详细过程***')
        result3 = metabolism_gm11(data, predict_num);
        result = (result1+result2+result3)/3;
        disp('对原始数据的拟合结果：')
        for i = 1:n
            disp(strcat(num2str(day(i)), ' ： ',num2str(data_hat(i))))
        end
        disp(strcat('传统GM(1,1)往后预测',num2str(predict_num),'期的得到的结果：'))
        for i = 1:predict_num
            disp(strcat(num2str(day(end)+i), ' ： ',num2str(result1(i))))
        end
        disp(strcat('新信息GM(1,1)往后预测',num2str(predict_num),'期的得到的结果：'))
        for i = 1:predict_num
            disp(strcat(num2str(day(end)+i), ' ： ',num2str(result2(i))))
        end
        disp(strcat('新陈代谢GM(1,1)往后预测',num2str(predict_num),'期的得到的结果：'))
        for i = 1:predict_num
            disp(strcat(num2str(day(end)+i), ' ： ',num2str(result3(i))))
        end
        disp(strcat('三种方法求平均得到的往后预测',num2str(predict_num),'期的得到的结果：'))
        for i = 1:predict_num
            disp(strcat(num2str(day(end)+i), ' ： ',num2str(result(i))))
        end
    end
    % 绘制相对残差
    figure(4)
    subplot(2,1,1)  
    plot(day(2:end), relative_residuals,'*-'); grid on;   
    legend('相对残差'); xlabel('年份序列');
    set(gca,'xtick',day(2:1:end))
    % 绘制级比偏差
    subplot(2,1,2)
    plot(day(2:end), eta,'o-'); grid on;   
    legend('级比偏差'); xlabel('年份序列');
    set(gca,'xtick',day(2:1:end))  
    disp(' ')
    disp('——————下面将输出对原数据拟合的评价———————')
    
    average_relative_residuals = mean(relative_residuals);  
    disp(strcat('平均相对残差为',num2str(average_relative_residuals)))
    if average_relative_residuals<0.1
        disp('残差检验的结果表明：该模型对原数据的拟合程度非常不错')
    elseif average_relative_residuals<0.2
        disp('残差检验的结果表明：该模型对原数据的拟合程度达到一般要求')
    else
        disp('残差检验的结果表明：该模型对原数据的拟合程度不太好')
    end
    
    average_eta = mean(eta);   
    disp(strcat('平均级比偏差为',num2str(average_eta)))
    if average_eta<0.1
        disp('级比偏差检验的结果表明：该模型对原数据的拟合程度非常不错')
    elseif average_eta<0.2
        disp('级比偏差检验的结果表明：该模型对原数据的拟合程度达到一般要求')
    else
        disp('级比偏差检验的结果表明：该模型对原数据的拟合程度不太好')
    end
    disp(' ')
    disp('————————————————分割线——————————————————')
    
    figure(5)  
    plot(day,data,'-o',  day,data_hat,'-*m',  day(end)+1:day(end)+predict_num,result,'-*b' );   grid on;
    hold on;
    plot([day(end),day(end)+1],[data(end),result(1)],'-*b')
    legend('原始数据','拟合数据','预测数据')  
    set(gca,'xtick',day(1):1:day(end)+predict_num)  % 设置x轴横坐标的间隔为1
    xlabel('年份序列');  ylabel('医疗卫生机构数');  
end

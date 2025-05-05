clear;clc  % ���������е����б������������ڵ����ݡ�
% ������к����ݵĳ�ʼֵ

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
]';  % ����ʼ����
day = (1:1:length(data))';  % �������ʾ������У���֪������1-7�죬Ԥ���8��
% ����ԭʼ���ݵ�ʱ������ͼ
figure(1);  %�Ի������б�ţ���ΪԤ������Ļ�����ֹһ��
plot(day,data,'o-'); 
grid on;  
set(gca,'xtick',day(1:1:end))  
xlabel('�������');  ylabel('ҽ������������');  % ����������ϱ�ǩ

% �������ݵĺϷ��Լ��顣
ERROR = 0;
if sum(data<0) > 0  % data<0����һ���߼�����(0-1���)�����������С��0��������λ��Ϊ1�����ԭʼ���ݾ�Ϊ�Ǹ�������ô����߼�������ȫΪ0����ͺ�Ҳ��0
    disp('��ɫԤ���ʱ�������в���ʹ�ø���')
    ERROR = 1;
end

n = length(data);  % ����ԭʼ���ݵĳ���
disp(strcat('ԭʼ���ݵĳ���Ϊ',num2str(n)))
if n<=3
    disp('������̫С���޷����л�ɫԤ��')
    ERROR = 1;
end

if n>10
    disp('������̫�࣬��ɫԤ���޷�������׼��Ԥ��')
end

if size(data,1) == 1
    data = data';
end
if size(day,1) == 1
    day = day';
end

% ����׼ָ�����ɼ���
if ERROR == 0   
    disp('���������������������������������ָ��ߡ�����������������������������������')
    disp('׼ָ�����ɼ���')
    x1 = cumsum(data);   
    rho = data(2:end) ./ x1(1:end-1);  
   
    figure(2)
    plot(day(2:end),rho,'o-',[day(2),day(end)],[0.5,0.5],'-'); grid on;
    text(day(end-1)+0.2,0.55,'�ٽ���')   
    set(gca,'xtick',day(2:1:end))  
    xlabel('�������');  ylabel('ԭʼ���ݵĹ⻬��');  % ����⻬��
    
    disp(strcat('ָ��1���⻬��С��0.5������ռ��Ϊ',num2str(100*sum(rho<0.5)/(n-1)),'%'))
    disp(strcat('ָ��2����ȥǰ����ʱ���⣬�⻬��С��0.5������ռ��Ϊ',num2str(100*sum(rho(3:end)<0.5)/(n-3)),'%'))
    disp('�ο���׼��ָ��1һ��Ҫ����60%, ָ��2Ҫ����90%������Ϊ�������ݿ���ͨ��������')
    Judge = input('����Ϊ����ͨ��׼ָ�����ɵļ����𣿿���ͨ��������1������������0��');
    if Judge == 0
        disp('��ɫԤ��ģ�Ͳ��ʺϴ�����')
        ERROR = 1;
    end
    disp('���������������������������������ָ��ߡ�����������������������������������')
end

% �������ݵ������������ݷ�Ϊѵ����������顣 
if ERROR == 0   
    if  n > 4  
        disp('��Ϊԭ���ݵ���������4���������ǿ��Խ��������Ϊѵ�����������')   
        if n > 7
            test_num = 3;
        else
            test_num = 2;
        end
        train_data = data(1:end-test_num);  
        disp('ѵ��������: ')
        disp(mat2str(train_data'))  
        test_data =  data(end-test_num+1:end); 
        disp('����������: ')
        disp(mat2str(test_data'))  
        disp('���������������������������������ָ��ߡ�����������������������������������')
        
        % ʹ�ô�ͳ��GM(1,1)ģ��
        disp(' ')
        disp('��ͳ��GM(1,1)ģ��Ԥ�����ϸ���̣�')
        result1 = gm11(train_data, test_num); 
        disp(' ')
        % ����Ϣ��GM(1,1)ģ��
        disp('�����ǽ�������Ϣ��GM(1,1)ģ��Ԥ�����ϸ���̣�')
        result2 = new_gm11(train_data, test_num); 
        disp(' ')
        % �³´�л��GM(1,1)ģ��
        disp('�����ǽ����³´�л��GM(1,1)ģ��Ԥ�����ϸ���̣�')
        result3 = metabolism_gm11(train_data, test_num); 
        
        disp(' ')
        disp('���������������������������������ָ��ߡ�����������������������������������')
        
        test_day = day(end-test_num+1:end);  
        figure(3)
        plot(test_day,test_data,'o-',test_day,result1,'*-',test_day,result2,'+-',test_day,result3,'x-'); grid on;
        set(gca,'xtick',day(end-test_num+1): 1 :day(end))  
        legend('���������ʵ����','��ͳGM(1,1)Ԥ����','����ϢGM(1,1)Ԥ����','�³´�лGM(1,1)Ԥ����')  
        xlabel('�������');  ylabel('ҽ������������');  
        
        SSE_1 = sum((test_data-result1).^2);
        SSE_2 = sum((test_data-result2).^2);
        SSE_3 = sum((test_data-result3).^2);
        disp(strcat('��ͳGM(1,1)����������Ԥ������ƽ����Ϊ',num2str(SSE_1)))
        disp(strcat('����ϢGM(1,1)����������Ԥ������ƽ����Ϊ',num2str(SSE_2)))
        disp(strcat('�³´�лGM(1,1)����������Ԥ������ƽ����Ϊ',num2str(SSE_3)))
        
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
        
        Model = {'��ͳGM(1,1)ģ��','����ϢGM(1,1)ģ��','�³´�лGM(1,1)ģ��'};
        disp(strcat('��Ϊ',Model(choose),'�����ƽ������С����������Ӧ��ѡ�������Ԥ��'))
        disp('���������������������������������ָ��ߡ�����������������������������������')
        
        predict_num = input('��������Ҫ������Ԥ��������� ');

        [result, data_hat, relative_residuals, eta] = gm11(data, predict_num);  
        
        if choose == 2
            result = new_gm11(data, predict_num);
        end
        if choose == 3
            result = metabolism_gm11(data, predict_num);
        end
        
        disp('���������������������������������ָ��ߡ�����������������������������������')
        disp('��ԭʼ���ݵ���Ͻ����')
        for i = 1:n
            disp(strcat(num2str(day(i)), ' �� ',num2str(data_hat(i))))
        end
        disp(strcat('����Ԥ��',num2str(predict_num),'�ڵĵõ��Ľ����'))
        for i = 1:predict_num
            disp(strcat(num2str(day(end)+i), ' �� ',num2str(result(i))))
        end
        
    else
        disp('��Ϊ����ֻ��4�ڣ��������ֱ�ӽ����ַ����Ľ����ƽ������~')
        predict_num = input('��������Ҫ������Ԥ��������� ');
        disp(' ')
        disp('***�����Ǵ�ͳ��GM(1,1)ģ��Ԥ�����ϸ����***')
        [result1, data_hat, relative_residuals, eta] = gm11(data, predict_num);
        disp(' ')
        disp('***�����ǽ�������Ϣ��GM(1,1)ģ��Ԥ�����ϸ����***')
        result2 = new_gm11(data, predict_num);
        disp(' ')
        disp('***�����ǽ����³´�л��GM(1,1)ģ��Ԥ�����ϸ����***')
        result3 = metabolism_gm11(data, predict_num);
        result = (result1+result2+result3)/3;
        disp('��ԭʼ���ݵ���Ͻ����')
        for i = 1:n
            disp(strcat(num2str(day(i)), ' �� ',num2str(data_hat(i))))
        end
        disp(strcat('��ͳGM(1,1)����Ԥ��',num2str(predict_num),'�ڵĵõ��Ľ����'))
        for i = 1:predict_num
            disp(strcat(num2str(day(end)+i), ' �� ',num2str(result1(i))))
        end
        disp(strcat('����ϢGM(1,1)����Ԥ��',num2str(predict_num),'�ڵĵõ��Ľ����'))
        for i = 1:predict_num
            disp(strcat(num2str(day(end)+i), ' �� ',num2str(result2(i))))
        end
        disp(strcat('�³´�лGM(1,1)����Ԥ��',num2str(predict_num),'�ڵĵõ��Ľ����'))
        for i = 1:predict_num
            disp(strcat(num2str(day(end)+i), ' �� ',num2str(result3(i))))
        end
        disp(strcat('���ַ�����ƽ���õ�������Ԥ��',num2str(predict_num),'�ڵĵõ��Ľ����'))
        for i = 1:predict_num
            disp(strcat(num2str(day(end)+i), ' �� ',num2str(result(i))))
        end
    end
    % ������Բв�
    figure(4)
    subplot(2,1,1)  
    plot(day(2:end), relative_residuals,'*-'); grid on;   
    legend('��Բв�'); xlabel('�������');
    set(gca,'xtick',day(2:1:end))
    % ���Ƽ���ƫ��
    subplot(2,1,2)
    plot(day(2:end), eta,'o-'); grid on;   
    legend('����ƫ��'); xlabel('�������');
    set(gca,'xtick',day(2:1:end))  
    disp(' ')
    disp('���������������潫�����ԭ������ϵ����ۡ�������������')
    
    average_relative_residuals = mean(relative_residuals);  
    disp(strcat('ƽ����Բв�Ϊ',num2str(average_relative_residuals)))
    if average_relative_residuals<0.1
        disp('�в����Ľ����������ģ�Ͷ�ԭ���ݵ���ϳ̶ȷǳ�����')
    elseif average_relative_residuals<0.2
        disp('�в����Ľ����������ģ�Ͷ�ԭ���ݵ���ϳ̶ȴﵽһ��Ҫ��')
    else
        disp('�в����Ľ����������ģ�Ͷ�ԭ���ݵ���ϳ̶Ȳ�̫��')
    end
    
    average_eta = mean(eta);   
    disp(strcat('ƽ������ƫ��Ϊ',num2str(average_eta)))
    if average_eta<0.1
        disp('����ƫ�����Ľ����������ģ�Ͷ�ԭ���ݵ���ϳ̶ȷǳ�����')
    elseif average_eta<0.2
        disp('����ƫ�����Ľ����������ģ�Ͷ�ԭ���ݵ���ϳ̶ȴﵽһ��Ҫ��')
    else
        disp('����ƫ�����Ľ����������ģ�Ͷ�ԭ���ݵ���ϳ̶Ȳ�̫��')
    end
    disp(' ')
    disp('���������������������������������ָ��ߡ�����������������������������������')
    
    figure(5)  
    plot(day,data,'-o',  day,data_hat,'-*m',  day(end)+1:day(end)+predict_num,result,'-*b' );   grid on;
    hold on;
    plot([day(end),day(end)+1],[data(end),result(1)],'-*b')
    legend('ԭʼ����','�������','Ԥ������')  
    set(gca,'xtick',day(1):1:day(end)+predict_num)  % ����x�������ļ��Ϊ1
    xlabel('�������');  ylabel('ҽ������������');  
end

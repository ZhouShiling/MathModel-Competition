% �������ã�ʹ���³´�л��GM(1,1)ģ�Ͷ����ݽ���Ԥ��
function [result] = metabolism_gm11(x0, predict_num)

    result = zeros(predict_num, 1);  % ��ʼ����������Ԥ��ֵ������
    for i = 1 : predict_num
        result(i) = gm11(x0, 1);  % ��Ԥ��һ�ڵĽ�����浽result��
        x0 = [x0(2:end); result(i)];  % ����x0����
    end
end
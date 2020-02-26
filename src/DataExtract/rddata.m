function  rddata( head,qrs,dat )
% This programm reads ECG data which are saved in format 212.
% (e.g., 100.dat from MIT-BIH-DB, cu01.dat from CU-DB,...)
% The data are displayed in a figure together with the annotations.
% The annotations are saved in the vector ANNOT, the corresponding
% times (in seconds) are saved in the vector ATRTIME.
% The annotations are saved as numbers, the meaning of the numbers can
% be found in the codetable "ecgcodes.h" available at www.physionet.org.
%
% ANNOT only contains the most important information, which is displayed
% with the program rdann (available on www.physionet.org) in the 3rd row.
% The 4th to 6th row are not saved in ANNOT.
%
%
%      created on Feb. 27, 2003 by
%      Robert Tratnig (Vorarlberg University of Applied Sciences)
%      (email: rtratnig@gmx.at),
%
%      algorithm is based on a program written by
%      Klaus Rheinberger (University of Innsbruck)
%      (email: klaus.rheinberger@uibk.ac.at)
%
%-------------------------------------------------------------------------


%------ SPECIFY DATA ------------------------------------------------------
%------ 指定数据文件 -------------------------------------------------------
PATH= 'C:\MATLAB\MIT-BIH\afpdb'; % 指定数据的储存路径
HEADERFILE= head;      % .hea 格式，头文件，可用记事本打开
DATAFILE= dat;         % .dat 格式，ECG 数据
SAMPLES2READ=230400;          % 指定需要读入的样本数
                            % 若.dat文件中存储有两个通道的信号:
                            % 则读入 2*SAMPLES2READ 个数据 

%------ LOAD HEADER DATA --------------------------------------------------
%------ 读入头文件数据 -----------------------------------------------------
%
% 示例：用记事本打开的117.hea 文件的数据
%
%      117 2 360 650000
%      117.dat 212 200 11 1024 839 31170 0 MLII
%      117.dat 212 200 11 1024 930 28083 0 V2
%      # 69 M 950 654 x2
%      # None
%
%-------------------------------------------------------------------------
fprintf(1,'\\n$> WORKING ON %s ...\n', HEADERFILE); % 在Matlab命令行窗口提示当前工作状态
% 
% 【注】函数 fprintf 的功能将格式化的数据写入到指定文件中。
% 表达式：count = fprintf(fid,format,A,...)
% 在字符串'format'的控制下，将矩阵A的实数数据进行格式化，并写入到文件对象fid中。该函数返回所写入数据的字节数 count。
% fid 是通过函数 fopen 获得的整型文件标识符。fid=1，表示标准输出（即输出到屏幕显示）；fid=2，表示标准偏差。
%
signalh= fullfile(PATH, HEADERFILE);    % 通过函数 fullfile 获得头文件的完整路径
fid1=fopen(signalh,'r');    % 打开头文件，其标识符为 fid1 ，属性为'r'--“只读”
if(fid1<=0)
    return;
end
z= fgetl(fid1);             % 读取头文件的第一行数据，字符串格式
A= sscanf(z, '%*s %d %d %d',[1,3]); % 按照格式 '%*s %d %d %d' 转换数据并存入矩阵 A 中
nosig= A(1);    % 信号通道数目
sfreq=A(2);     % 数据采样频率
clear A;        % 清空矩阵 A ，准备获取下一行数据
for k=1:nosig           % 读取每个通道信号的数据信息
    z= fgetl(fid1);
    A= sscanf(z, '%*s %d %d %d %d %d',[1,5]);
    dformat(k)= A(1);           % 信号格式; 这里只允许为 212 格式，新数据集换成16
    gain(k) = A(2);                    % 每 mV 包含的整数个数
    bitres(k)= A(3);            % 采样精度（位分辨率）
    zerovalue(k)= A(4);         % ECG 信号零点相应的整数值
    firstvalue(k)= A(5);        % 信号的第一个整数值 (用于偏差测试)
end;
fclose(fid1);
clear A;

%------ LOAD BINARY DATA --------------------------------------------------
%------ 读取 ECG 信号二值数据 ----------------------------------------------
%
signald= fullfile(PATH, DATAFILE);            % 读入 16 格式的 ECG 信号数据
fid2=fopen(signald,'r');
A= fread(fid2, [4, SAMPLES2READ], 'uint8')';  % 四个字节表示2个数
fclose(fid2);


% 通过一系列的移位（bitshift）、位与（bitand）运算，将信号由二值数据转换为十进制数
M1H= bitshift(A(:,2), 8);        %第二个字节左移8位
M2H= bitshift(A(:,4), 8);          %第四列左移8位
PR1 = bitshift(bitand(A(:,2),128),9);
PR2 = bitshift(bitand(A(:,4),128),9);
M( : , 1)= M1H+ A(:,1)-PR1;
M( : , 2)= M2H+ A(:,3)-PR2;

if M(1,:) ~= firstvalue, error('inconsistency in the first bit values'); end;
switch nosig
case 2
    M( : , 1)= (M( : , 1)- zerovalue(1))/gain(1);
    M( : , 2)= (M( : , 2)- zerovalue(2))/gain(2);
    TIME=(0:(SAMPLES2READ-1))/sfreq;
case 1
    M( : , 1)= (M( : , 1)- zerovalue(1));
    M( : , 2)= (M( : , 2)- zerovalue(1));
    M=M';
    M(1)=[];
    sM=size(M);
    sM=sM(2)+1;
    M(sM)=0;
    M=M';
    M=M/gain(1);
    TIME=(0:2*(SAMPLES2READ)-1)/sfreq;
otherwise  % this case did not appear up to now!
    % here M has to be sorted!!!
    disp('Sorting algorithm for more than 2 signals not programmed yet!');
end;
clear A M1H M2H PRR PRL;
fprintf(1,'\\n$> LOADING DATA FINISHED \n');

%%下采样,下采样之后按照振幅划分等级，先找到最大值和最小值
down_sample = dyaddown(M(:,1),1)
max_1 = max(down_sample);
min_1 = min(down_sample);
dist = max_1-min_1;
sig = [];
%%按照振幅符号化
for i=1:size(down_sample)
    new_dist = down_sample(i)-min_1;
    per = new_dist/dist;
    if 0<=per && per<0.2
        sig=[sig,1];
    elseif 0.2<=per && per<0.4
        sig = [sig,2];
    elseif 0.4<=per && per<0.6
        sig = [sig,3];
    elseif 0.6<=per && per<0.8
        sig = [sig,4];
    else 
        sig = [sig,5];
    end
end
%%截取一分钟长度的信息
fid=fopen('C:\ECGtest\info\pc_y_train.txt','a');    % 打开头文件，其标识符为 fid1 ，属性为'r'--“只读”
c = '*';
for i=1:64*60:size(down_sample)
    if (i+64*60)>size(down_sample)
        break;
    end
    for k=i:i+64*60
        fprintf(fid,'%d',sig(k));
    end
    fprintf(fid,'%c\n',c);
end
fclose(fid);
%{
%------ DISPLAY DATA ------------------------------------------------------
RR=[];
last=find(TIME==ATRTIME(3));
%%记录第3个的位置，进行做差
%%RR间隔留下数字 记录前后
for k=4:size(ATRTIME,1)
    gg=find(TIME==ATRTIME(k));
    RR=[RR,TIME(gg)-TIME(last)];
    last=gg;
end
%%此时RR中表示的是RR之间的间隔时间


%%提取R旁边数据信息，每个波群（含波峰）共250个点
ECG_ALL = [];
ECG_ALL_TEMP=[];
for  my_gg = 4 :4: size(ATRTIME,1)-4
    for my_k = find(TIME == ATRTIME(my_gg))-75 : find(TIME == ATRTIME(my_gg+3))+124
        ECG_ALL_TEMP = [ECG_ALL_TEMP , M(my_k,1)];
    end
end

for k=2:length(ECG_ALL_TEMP)
    if ECG_ALL_TEMP(k)>ECG_ALL_TEMP(k-1)
        ECG_ALL = [ECG_ALL,1];
    else ECG_ALL = [ECG_ALL,0];
    end
end
ECG_ALL_chafen=[];    

%%将差分数组写入文件
fid=fopen('C:\ECGtest\info\arr_train_1.txt','a');    % 打开头文件，其标识符为 fid1 ，属性为'r'--“只读”
a='P';
b='N';
c='*';
my_gg=1;
for i=1:1000:length(ECG_ALL)-1000
    
    if i+1000<=length(ECG_ALL)-1000
        for g = 1:4
            for ggg = i+(g-1)*250:i+g*250
                fprintf(fid,'%d',ECG_ALL(ggg));
            end
                fprintf(fid,'%c',a);
                fprintf(fid,'%f',RR(my_gg));
                fprintf(fid,'%c',b);
                fprintf(fid,'%f',RR(my_gg+1));
                fprintf(fid,'%c\n',c);
        end
    end
    my_gg = my_gg+1;
end

fclose(fid);
%}
% -------------------------------------------------------------------------
fprintf(1,'\\n$> ALL FINISHED \n');
end
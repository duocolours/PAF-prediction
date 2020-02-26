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
%------ ָ�������ļ� -------------------------------------------------------
PATH= 'C:\MATLAB\MIT-BIH\afpdb'; % ָ�����ݵĴ���·��
HEADERFILE= head;      % .hea ��ʽ��ͷ�ļ������ü��±���
DATAFILE= dat;         % .dat ��ʽ��ECG ����
SAMPLES2READ=230400;          % ָ����Ҫ�����������
                            % ��.dat�ļ��д洢������ͨ�����ź�:
                            % ����� 2*SAMPLES2READ ������ 

%------ LOAD HEADER DATA --------------------------------------------------
%------ ����ͷ�ļ����� -----------------------------------------------------
%
% ʾ�����ü��±��򿪵�117.hea �ļ�������
%
%      117 2 360 650000
%      117.dat 212 200 11 1024 839 31170 0 MLII
%      117.dat 212 200 11 1024 930 28083 0 V2
%      # 69 M 950 654 x2
%      # None
%
%-------------------------------------------------------------------------
fprintf(1,'\\n$> WORKING ON %s ...\n', HEADERFILE); % ��Matlab�����д�����ʾ��ǰ����״̬
% 
% ��ע������ fprintf �Ĺ��ܽ���ʽ��������д�뵽ָ���ļ��С�
% ���ʽ��count = fprintf(fid,format,A,...)
% ���ַ���'format'�Ŀ����£�������A��ʵ�����ݽ��и�ʽ������д�뵽�ļ�����fid�С��ú���������д�����ݵ��ֽ��� count��
% fid ��ͨ������ fopen ��õ������ļ���ʶ����fid=1����ʾ��׼��������������Ļ��ʾ����fid=2����ʾ��׼ƫ�
%
signalh= fullfile(PATH, HEADERFILE);    % ͨ������ fullfile ���ͷ�ļ�������·��
fid1=fopen(signalh,'r');    % ��ͷ�ļ������ʶ��Ϊ fid1 ������Ϊ'r'--��ֻ����
if(fid1<=0)
    return;
end
z= fgetl(fid1);             % ��ȡͷ�ļ��ĵ�һ�����ݣ��ַ�����ʽ
A= sscanf(z, '%*s %d %d %d',[1,3]); % ���ո�ʽ '%*s %d %d %d' ת�����ݲ�������� A ��
nosig= A(1);    % �ź�ͨ����Ŀ
sfreq=A(2);     % ���ݲ���Ƶ��
clear A;        % ��վ��� A ��׼����ȡ��һ������
for k=1:nosig           % ��ȡÿ��ͨ���źŵ�������Ϣ
    z= fgetl(fid1);
    A= sscanf(z, '%*s %d %d %d %d %d',[1,5]);
    dformat(k)= A(1);           % �źŸ�ʽ; ����ֻ����Ϊ 212 ��ʽ�������ݼ�����16
    gain(k) = A(2);                    % ÿ mV ��������������
    bitres(k)= A(3);            % �������ȣ�λ�ֱ��ʣ�
    zerovalue(k)= A(4);         % ECG �ź������Ӧ������ֵ
    firstvalue(k)= A(5);        % �źŵĵ�һ������ֵ (����ƫ�����)
end;
fclose(fid1);
clear A;

%------ LOAD BINARY DATA --------------------------------------------------
%------ ��ȡ ECG �źŶ�ֵ���� ----------------------------------------------
%
signald= fullfile(PATH, DATAFILE);            % ���� 16 ��ʽ�� ECG �ź�����
fid2=fopen(signald,'r');
A= fread(fid2, [4, SAMPLES2READ], 'uint8')';  % �ĸ��ֽڱ�ʾ2����
fclose(fid2);


% ͨ��һϵ�е���λ��bitshift����λ�루bitand�����㣬���ź��ɶ�ֵ����ת��Ϊʮ������
M1H= bitshift(A(:,2), 8);        %�ڶ����ֽ�����8λ
M2H= bitshift(A(:,4), 8);          %����������8λ
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

%%�²���,�²���֮����������ֵȼ������ҵ����ֵ����Сֵ
down_sample = dyaddown(M(:,1),1)
max_1 = max(down_sample);
min_1 = min(down_sample);
dist = max_1-min_1;
sig = [];
%%����������Ż�
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
%%��ȡһ���ӳ��ȵ���Ϣ
fid=fopen('C:\ECGtest\info\pc_y_train.txt','a');    % ��ͷ�ļ������ʶ��Ϊ fid1 ������Ϊ'r'--��ֻ����
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
%%��¼��3����λ�ã���������
%%RR����������� ��¼ǰ��
for k=4:size(ATRTIME,1)
    gg=find(TIME==ATRTIME(k));
    RR=[RR,TIME(gg)-TIME(last)];
    last=gg;
end
%%��ʱRR�б�ʾ����RR֮��ļ��ʱ��


%%��ȡR�Ա�������Ϣ��ÿ����Ⱥ�������壩��250����
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

%%���������д���ļ�
fid=fopen('C:\ECGtest\info\arr_train_1.txt','a');    % ��ͷ�ļ������ʶ��Ϊ fid1 ������Ϊ'r'--��ֻ����
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
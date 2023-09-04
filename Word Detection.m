%%
%sampling
%part a

clc;
clear;
close all;

Fs=100;
t=0:1/Fs:10-1/Fs;
x=cos(100*t)+2*cos(200*t);

HalfBandFFT1(x,100);
HalfBandFFT2(x,100);

%%
%sampling
%c
clc;
clear;
close all;

T1=1/6;
T2=1/2;
t=0:0.01:2*pi;
wmax1=pi/2;
wmax2=wmax1*1/T1;
ws=2*pi*1/T2;
wmin=(ws-wmax2)*T2;
wmax=wmax2*T2;
w1=floor(wmax*100);
w2=floor((2*pi-wmin)*100);
t1=linspace(0,1,w1);
t2=linspace(0,1,w2);
l1=-1*t1+1;
l2=t2;
diagram1=zeros(1,length(t));
diagram2=zeros(1,length(t));
diagram1(1:w1)=l1;
diagram2(end-w2+1:end)=l2;
figure;
plot(t,diagram1,'linewidth',2,'color','r');
hold on;
plot(t,diagram2,'linewidth',2,'color','r');
xlim([0,2*pi]);
title('a single period of a sample signal');


%%
%%EEG signals
clc;
clear;
close all;
EEG1=load('SubjectData1.mat');
EEG2=load('SubjectData2.mat');
EEG3=load('SubjectData3.mat');
EEG5=load('SubjectData5.mat');
EEG6=load('SubjectData6.mat');
EEG7=load('SubjectData7.mat');
EEG8=load('SubjectData8.mat');
EEG9=load('SubjectData9.mat');

%%
%%EEG signals
%part a
clc;
close all;
Fs=1/abs(EEG1.train(1,1)-EEG1.train(1,2));

%%
%%EEG signals
%part c
clc;
close all;

%first experiment
for i=2:1:9
   HalfBandFFT2(EEG1.train(i,:),Fs) ;
end
%%
%second experiment
clc;
close all;

for i=2:1:9
   HalfBandFFT2(EEG2.train(i,:),Fs) ;
end
%%
clc;
close all;

%third experiment
for i=2:1:9
   HalfBandFFT2(EEG3.train(i,:),Fs) ;
end

%%
clc;
close all;

%fifth experiment
for i=2:1:9
   HalfBandFFT2(EEG5.train(i,:),Fs) ;
end

%%
clc;
close all;

%sixth experiment
for i=2:1:9
   HalfBandFFT2(EEG6.train(i,:),Fs) ;
end

%%
clc;
close all;

%seventh experiment
for i=2:1:9
   HalfBandFFT2(EEG7.train(i,:),Fs) ;
end

%%
clc;
close all;

%eightth experiment
for i=2:1:9
   HalfBandFFT2(EEG8.train(i,:),Fs) ;
end

%%
clc;
close all;

%nineth experiment
for i=2:1:9
   HalfBandFFT2(EEG9.train(i,:),Fs) ;
end

%%
%%EEG signals
%part d

%first experiment
clc;
close all;

frequencies1=zeros(1,8);
for i=2:1:9
figure;
obw(EEG1.train(i,:),Fs);
frequencies1(i-1)=obw(EEG1.train(i,:),Fs);
end

%%
%second experiment
clc;
close all;

frequencies2=zeros(1,8);
for i=2:1:9
figure;
obw(EEG2.train(i,:),Fs);
frequencies2(i-1)=obw(EEG2.train(i,:),Fs);
end

%%
%third experiment
clc;
close all;

frequencies3=zeros(1,8);
for i=2:1:9
figure;
obw(EEG3.train(i,:),Fs);
frequencies3(i-1)=obw(EEG3.train(i,:),Fs);
end

%%
%fifth experiment
clc;
close all;

frequencies5=zeros(1,8);
for i=2:1:9
figure;
obw(EEG5.train(i,:),Fs);
frequencies5(i-1)=obw(EEG5.train(i,:),Fs);
end

%%
%sixth experiment
clc;
close all;

frequencies6=zeros(1,8);
for i=2:1:9
figure;
obw(EEG6.train(i,:),Fs);
frequencies6(i-1)=obw(EEG6.train(i,:),Fs);
end

%%
%seventh experiment
clc;
close all;

frequencies7=zeros(1,8);
for i=2:1:9
figure;
obw(EEG7.train(i,:),Fs);
frequencies7(i-1)=obw(EEG7.train(i,:),Fs);
end

%%
%eighth experiment
clc;
close all;

frequencies8=zeros(1,8);
for i=2:1:9
figure;
obw(EEG8.train(i,:),Fs);
frequencies8(i-1)=obw(EEG8.train(i,:),Fs);
end

%%
%nineth experiment
clc;
close all;

frequencies9=zeros(1,8);
for i=2:1:9
figure;
obw(EEG9.train(i,:),Fs);
frequencies9(i-1)=obw(EEG9.train(i,:),Fs);
end

%%
%%EEG signals
%part g

%first experiment

clc;
close all;
s1=EEG1.train(2:9,:);
a1=mean(s1,2);
b1=repmat(a1,[1,length(s1(1,:))]);
s1=s1-b1;
load('bandpassfilter');
N=100000;
gd=groupdelay(Num,N);

ss1=EEG1.test(2:9,:);
aa1=mean(ss1,2);
bb1=repmat(aa1,[1,length(ss1(1,:))]);
ss1=ss1-bb1;

for i=1:1:8
filteredEEGtrain1(i,:)=zphasefilter(Num,s1(i,:),gd,Fs);
end

for i=1:1:8
filteredEEGtest1(i,:)=zphasefilter(Num,ss1(i,:),gd,Fs);
end

%%
%second experiment

clc;
close all;
s2=EEG2.train(2:9,:);
a2=mean(s2,2);
b2=repmat(a2,[1,length(s2(1,:))]);
s2=s2-b2;

ss2=EEG2.test(2:9,:);
aa2=mean(ss2,2);
bb2=repmat(aa2,[1,length(ss2(1,:))]);
ss2=ss2-bb2;

for i=1:1:8
filteredEEGtrain2(i,:)=filtfilt(Num,1,s2(i,:));
end

for i=1:1:8
filteredEEGtest2(i,:)=filtfilt(Num,1,ss2(i,:));
end

%%
%third experiment

clc;
close all;
s3=EEG3.train(2:9,:);
a3=mean(s3,2);
b3=repmat(a3,[1,length(s3(1,:))]);
s3=s3-b3;

ss3=EEG3.test(2:9,:);
aa3=mean(ss3,2);
bb3=repmat(aa3,[1,length(ss3(1,:))]);
ss3=ss3-bb3;

for i=1:1:8
filteredEEGtrain3(i,:)=filtfilt(Num,1,s3(i,:));
end

for i=1:1:8
filteredEEGtest3(i,:)=filtfilt(Num,1,ss3(i,:));
end

%%
%fifth experiment

clc;
close all;
s5=EEG5.train(2:9,:);
a5=mean(s5,2);
b5=repmat(a5,[1,length(s5(1,:))]);
s5=s5-b5;

ss5=EEG5.test(2:9,:);
aa5=mean(ss5,2);
bb5=repmat(aa5,[1,length(ss5(1,:))]);
ss5=ss5-bb5;

for i=1:1:8
filteredEEGtrain5(i,:)=filtfilt(Num,1,s5(i,:));
end

for i=1:1:8
filteredEEGtest5(i,:)=filtfilt(Num,1,ss5(i,:));
end

%%
%sixth experiment

clc;
close all;
s6=EEG6.train(2:9,:);
a6=mean(s6,2);
b6=repmat(a6,[1,length(s6(1,:))]);
s6=s6-b6;

ss6=EEG6.test(2:9,:);
aa6=mean(ss6,2);
bb6=repmat(aa6,[1,length(ss6(1,:))]);
ss6=ss6-bb6;

for i=1:1:8
filteredEEGtrain6(i,:)=filtfilt(Num,1,s6(i,:));
end

for i=1:1:8
filteredEEGtest6(i,:)=filtfilt(Num,1,ss6(i,:));
end

%%
%seventh experiment

clc;
close all;
s7=EEG7.train(2:9,:);
a7=mean(s7,2);
b7=repmat(a7,[1,length(s7(1,:))]);
s7=s7-b7;

ss7=EEG7.test(2:9,:);
aa7=mean(ss7,2);
bb7=repmat(aa7,[1,length(ss7(1,:))]);
ss7=ss7-bb7;

for i=1:1:8
filteredEEGtrain7(i,:)=filtfilt(Num,1,s7(i,:));
end

for i=1:1:8
filteredEEGtest7(i,:)=filtfilt(Num,1,ss7(i,:));
end

%%
%eighth experiment

clc;
close all;
s8=EEG8.train(2:9,:);
a8=mean(s8,2);
b8=repmat(a8,[1,length(s8(1,:))]);
s8=s8-b8;

ss8=EEG8.test(2:9,:);
aa8=mean(ss8,2);
bb8=repmat(aa8,[1,length(ss8(1,:))]);
ss8=ss8-bb8;

for i=1:1:8
filteredEEGtrain8(i,:)=filtfilt(Num,1,s8(i,:));
end

for i=1:1:8
filteredEEGtest8(i,:)=filtfilt(Num,1,ss8(i,:));
end

%%
%nineth experiment

clc;
close all;
s9=EEG9.train(2:9,:);
a9=mean(s9,2);
b9=repmat(a9,[1,length(s9(1,:))]);
s9=s9-b9;

ss9=EEG9.test(2:9,:);
aa9=mean(ss9,2);
bb9=repmat(aa9,[1,length(ss9(1,:))]);
ss9=ss9-bb9;

for i=1:1:8
filteredEEGtrain9(i,:)=filtfilt(Num,1,s9(i,:));
end

for i=1:1:8
filteredEEGtest9(i,:)=filtfilt(Num,1,ss9(i,:));
end


%%
%%EEG signals
%part h
clc;
close all;

%first experiment
for i=1:1:8
finaltrain1(i,:)=downsampler(filteredEEGtrain1(i,:),2);
end

for i=1:1:8
finaltest1(i,:)=downsampler(filteredEEGtest1(i,:),2);
end

%second experiment
for i=1:1:8
finaltrain2(i,:)=downsampler(filteredEEGtrain2(i,:),2);
end

for i=1:1:8
finaltest2(i,:)=downsampler(filteredEEGtest2(i,:),2);
end

%third experiment
for i=1:1:8
finaltrain3(i,:)=downsampler(filteredEEGtrain3(i,:),2);
end

for i=1:1:8
finaltest3(i,:)=downsampler(filteredEEGtest3(i,:),2);
end

%fifth experiment
for i=1:1:8
finaltrain5(i,:)=downsampler(filteredEEGtrain5(i,:),2);
end

for i=1:1:8
finaltest5(i,:)=downsampler(filteredEEGtest5(i,:),2);
end

%sixth experiment
for i=1:1:8
finaltrain6(i,:)=downsampler(filteredEEGtrain6(i,:),2);
end

for i=1:1:8
finaltest6(i,:)=downsampler(filteredEEGtest6(i,:),2);
end

%seventh experiment
for i=1:1:8
finaltrain7(i,:)=downsampler(filteredEEGtrain7(i,:),2);
end

for i=1:1:8
finaltest7(i,:)=downsampler(filteredEEGtest7(i,:),2);
end

%eighth experiment
for i=1:1:8
finaltrain8(i,:)=downsampler(filteredEEGtrain8(i,:),2);
end

for i=1:1:8
finaltest8(i,:)=downsampler(filteredEEGtest8(i,:),2);
end

%nineth experiment
for i=1:1:8
finaltrain9(i,:)=downsampler(filteredEEGtrain9(i,:),2);
end

for i=1:1:8
finaltest9(i,:)=downsampler(filteredEEGtest9(i,:),2);
end

%%
%%EEG signals
%part I,J
clc;
close all;
backwardsample=0.2;
forwardsample=0.8;

%first experiment
stimulionset=downsampler(EEG1.train(10,:),2);
inputsignal=finaltrain1;
epochtrain1=epochconsructor(inputsignal,backwardsample,forwardsample,stimulionset,Fs/2);
save('epochData1','epochtrain1');

stimulionset=downsampler(EEG1.test(10,:),2);
inputsignal=finaltest1;
epochtest1=epochconsructor(inputsignal,backwardsample,forwardsample,stimulionset,Fs/2);
save('epochtest1','epochtest1');

%second experiment
stimulionset=downsampler(EEG2.train(10,:),2);
inputsignal=finaltrain2;
epochtrain2=epochconsructor(inputsignal,backwardsample,forwardsample,stimulionset,Fs/2);
save('epochData2','epochtrain2');

stimulionset=downsampler(EEG2.test(10,:),2);
inputsignal=finaltest2;
epochtest2=epochconsructor(inputsignal,backwardsample,forwardsample,stimulionset,Fs/2);
save('epochtest2','epochtest2');


%third experiment
stimulionset=downsampler(EEG3.train(10,:),2);
inputsignal=finaltrain3;
epochtrain3=epochconsructor(inputsignal,backwardsample,forwardsample,stimulionset,Fs/2);
save('epochData3','epochtrain3');

stimulionset=downsampler(EEG3.test(10,:),2);
inputsignal=finaltest3;
epochtest3=epochconsructor(inputsignal,backwardsample,forwardsample,stimulionset,Fs/2);
save('epochtest3','epochtest3');

%fifth experiment
stimulionset=downsampler(EEG5.train(10,:),2);
inputsignal=finaltrain5;
epochtrain5=epochconsructor(inputsignal,backwardsample,forwardsample,stimulionset,Fs/2);
save('epochData5','epochtrain5');

stimulionset=downsampler(EEG5.test(10,:),2);
inputsignal=finaltest5;
epochtest5=epochconsructor(inputsignal,backwardsample,forwardsample,stimulionset,Fs/2);
save('epochtest5','epochtest5');

%sixth experiment
stimulionset=downsampler(EEG6.train(10,:),2);
inputsignal=finaltrain6;
epochtrain6=epochconsructor(inputsignal,backwardsample,forwardsample,stimulionset,Fs/2);
save('epochData6','epochtrain6');

stimulionset=downsampler(EEG6.test(10,:),2);
inputsignal=finaltest6;
epochtest6=epochconsructor(inputsignal,backwardsample,forwardsample,stimulionset,Fs/2);
save('epochtest6','epochtest6');

%seventh experiment
stimulionset=downsampler(EEG7.train(10,:),2);
inputsignal=finaltrain7;
epochtrain7=epochconsructor(inputsignal,backwardsample,forwardsample,stimulionset,Fs/2);
save('epochData7','epochtrain7');

stimulionset=downsampler(EEG7.test(10,:),2);
inputsignal=finaltest7;
epochtest7=epochconsructor(inputsignal,backwardsample,forwardsample,stimulionset,Fs/2);
save('epochtest7','epochtest7');

%eighth experiment
stimulionset=downsampler(EEG8.train(10,:),2);
inputsignal=finaltrain8;
epochtrain8=epochconsructor(inputsignal,backwardsample,forwardsample,stimulionset,Fs/2);
save('epochData8','epochtrain8');

stimulionset=downsampler(EEG8.test(10,:),2);
inputsignal=finaltest8;
epochtest8=epochconsructor(inputsignal,backwardsample,forwardsample,stimulionset,Fs/2);
save('epochtest8','epochtest8');

%nineth experiment
stimulionset=downsampler(EEG9.train(10,:),2);
inputsignal=finaltrain9;
epochtrain9=epochconsructor(inputsignal,backwardsample,forwardsample,stimulionset,Fs/2);
save('epochData9','epochtrain9');

stimulionset=downsampler(EEG9.test(10,:),2);
inputsignal=finaltest9;
epochtest9=epochconsructor(inputsignal,backwardsample,forwardsample,stimulionset,Fs/2);
save('epochtest9','epochtest9');


%%
%filter,group delay
clc;
close all;
clear;
load('bandpassfilter');
[gd,w]=grpdelay(Num,1,1000000);
figure;
plot(w,gd);
title('grpdelay for bandpass filter');
N=1000000;
gd=groupdelay(Num,N);

load('bandPass_clustring');
[gd,w]=grpdelay(Num,1,1000000);
figure;
plot(w,gd);
title('grpdelay for bandpass clustering');
N=1000000;
gd=groupdelay(Num,N);

load('BPfilter');
[gd,w]=grpdelay(Num,1,1000000);
figure;
plot(w,gd);
title('grpdelay for BPfilter');
N=1000000;
gd=groupdelay(Num,N);

%%
%filter,zphasefilter
clc;
clear;
close all;

EEG=load('SubjectData1');
load('bandpassfilter');
x=EEG.train(2,:);
N=1000000;
gd=groupdelay(Num,N);
s1=zphasefilter(Num,x,gd,256);
s2=filtfilt(Num,1,x);

figure;
plot(s1);
hold on;
plot(s2);
xlim([450 800]);


%%
%clustring
clear all;
clc;
close all;
txt='64channeldata.mat';
n=63;
data=load(txt);
cell = struct2cell(data);
data = double(cell2mat(cell));

dn=5;
newData=zeros(63,1800*44/dn);
a=load('bandPass_clustring');
cell = struct2cell(a);
a = double(cell2mat(cell));
for i=1:1:n
z=reshape(data(i,:,:),1, 1800*44);
z=z-mean(z);
%%obw(z,600);
z=filtfilt(a,1,z);
%HalfBandFFT2(z,600);
z=downsample(z,dn);
newData(i,:)=z;

end
rxy=abs(correlationMatrix(newData,63));
rrr=rxy;
distance=1-rxy;
dd=distance;
[G Dmatrix]=UPGMA(8,distance);
channel=diag(G);
r=find(~(isnan(channel)));
n=0;
channelnum=zeros(1,63);
for i=1:1:size(r)
nChannel=G((r(i)),:);
nChannel=find(nChannel~=0)
for k=1:1:size(nChannel,2)
    channelnum(1,nChannel(k))=i*100;
end
n=size(nChannel,2)+n;
end
channelnum=channelnum';
figure
ch_list = {'AFZ','FP1','FP2','AF3','AF4','F7','F3','FZ','F4','F8','FC5','FC1','FC2','FC6','T7','C3','CZ','C4','T8','CP5','CP1',...
'CP2','CP6','P7','P3','PZ','P4','P8','PO3','PO4','O1','O2','FT10','AF7','AF8','F5','F1','F2','F6','FT7','FC3','FCZ','FC4',...
'FT8','C5','C1','C2','C6','TP7','CP3','CPZ','CP4','TP8','P5','P1','P2','P6','PO7','POZ','PO8','OZ','TP9','TP10'};
h = plot_topography(ch_list, channelnum, false);
title('64 channel');
h=load('epochData1.mat');
cell = struct2cell(h);
h = double(cell2mat(cell));
n1=size(h,1);
n2=size(h,2);
n3=size(h,3);
new=zeros(n1,n2*n3);
for i=1:1:n1
t=reshape(h(i,:,:),1, n2*n3);
new(i,:)=t;
end
rxy=abs(correlationMatrix(new,n1));
distance=1-rxy;
[G]=UPGMA(4,distance);
channel=diag(G);
r=find(~(isnan(channel)));
channelnum=zeros(1,8);
for i=1:1:size(r)
nChannel=G((r(i)),:);
nChannel=find(nChannel~=0)
for k=1:1:size(nChannel,2)
    channelnum(1,nChannel(k))=i*100;
end
n=size(nChannel,2)+n;
end
channelnum=channelnum';
ch_list = {'FZ','CZ','PZ','PO7','P3','PO8','OZ','P4'};
figure
h = plot_topography(ch_list, channelnum, false);
title('first clustring 8 channel');
ch_list ={'FZ','CZ','P3','PZ','P4','PO7','PO8','OZ'};
figure
h = plot_topography(ch_list, channelnum, false);
title('second clustring 8 channel');
%%
%detecting word and machine lening
clc;
clear all;
close all;
subject1=IndexExtraction('SubjectData2.mat');
subject9=IndexExtraction('SubjectData9.mat');
cost=[0 1;4 0];
[q9 res]=detecting(subject9,'epochData9','epochData9','fitcsvm','tr',15,2,cost,0);
word_trian_RC=detectRC(q9)
[q1 res]=detecting(subject1,'epochData2','epochData2','fitcsvm','tr',15,2,cost,0);
word_train_SC=detectSC(q1)
cost=[0 10;0.5 0];
[q11 res]=detecting(subject1,'epochData2','epochData2','fitclda','tr',15,2,cost,0);
word_train_SC_lda=detectSC(q11)
cost=[0 10;0.5 0];
[q111 res]=detecting(subject1,'epochtest2','epochData2','fitcsvm','te',3,2,cost,1);
word_train_SC_test=detectSC(q111)
[q999 res]=detecting(subject9,'epochtest9','epochData9','fitcsvm','te',9,2,cost,0);
word_trian_RC_trian=detectRC(q999)

cost=[0 0.5;10 0];%for num=0
%cost=[0 10;0.5 0];%for num=2
%q2=detecting(subject,'epochtest6','fitclda','te',10,0,cost);

%word=detectSC(q)
%word=detectRC(q)
%%
%functions

function ss=zphasefilter(h,s,gd,Fs)
a=zeros(1,ceil(gd(1)));
b=ceil(ceil(gd(1))/length(s));
for i=1:1:b
if i==1
s3=[s s];
end
if i~=1
s3=[s3 s];
end

end

filtereds3=filter(h,1,s3);
ss=filtereds3(ceil(gd(1)+1):ceil(gd(1)+1)+length(s)-1);

end

function gd=groupdelay(h,N)

Hw=fft(h,N);
n=0:1:length(h)-1;
hh=n.*h;
dHw=fft(hh,N);
y=dHw./Hw;
gd=real(y);
w=linspace(0,pi,length(gd));
figure;
plot(w,gd);

end

function y=downsampler(x,n)

for i=1:1:floor(length(x)/n)
y(i)=x(n*i);
end
    
end

function epoch=epochconsructor(inputsignal,backwardsample,forwardsample,stimulionset,Fs)
n1=backwardsample;
n2=forwardsample;
n1=floor(n1.*Fs);
n2=floor(n2.*Fs);
a=find(stimulionset~=0);
b=a;
for i=1:1:length(a)
    if i~=1
   if(a(i)-a(i-1)>1)
    b(i)=-1;  
   end
    end
    if i==1
       b(i)=-1; 
    end
end

c=find(b==-1);
d=a(c);

for i=1:1:length(d)
   epoch(1,i,:)=inputsignal(1,d(i)-n1:d(i)+n2);
   epoch(2,i,:)=inputsignal(2,d(i)-n1:d(i)+n2);
   epoch(3,i,:)=inputsignal(3,d(i)-n1:d(i)+n2);
   epoch(4,i,:)=inputsignal(4,d(i)-n1:d(i)+n2);
   epoch(5,i,:)=inputsignal(5,d(i)-n1:d(i)+n2);
   epoch(6,i,:)=inputsignal(6,d(i)-n1:d(i)+n2);
   epoch(7,i,:)=inputsignal(7,d(i)-n1:d(i)+n2);
   epoch(8,i,:)=inputsignal(8,d(i)-n1:d(i)+n2);
end

end

function HalfBandFFT1(x,Fs)

l=length(x);
fftx=abs(fft(x))/(l/2);
f=(0:floor(l/2)-1)/l*Fs*2*pi;
fftt=fftx(1:floor(end/2));
f=f/f(end)*pi;
figure;
plot(f,fftt,'linewidth',2,'color','r');

end

function HalfBandFFT2(x,Fs)

l=length(x);
fftx=abs(fft(x))/(l/2);
f=(0:floor(l/2)-1)/l*Fs;
fftt=fftx(1:floor(end/2));
figure;
plot(f,fftt,'linewidth',2,'color','r');

end

function corr=correlation(X,Y)
Z=X.*Y;
U=X.^2;
K=Y.^2;
sumZ=sum(sum(Z));
sumU=sum(sum(U));
sumK=sum(sum(K));
S=sqrt(sumU.*sumK);
corr=sumZ./S;
end
function rxy=correlationMatrix(data,numOfChanell)
num=numOfChanell;
rxy=zeros(num,num);
for i=1:1:num
    for j=1:1:num
        rxy(i,j)=correlation(data(i,:),data(j,:));
    end
end
end
function [Group Dmatrix]=UPGMA(numOfStage,Dmatrix)
n=0;
siz=size(Dmatrix,1);
Group=eye(siz,siz);
while (n~=numOfStage)

[row,column]=find(Dmatrix==0);
for i=1:1:size(row)
    Dmatrix(row(i),column(i))=2;
end

sorted=sort(Dmatrix(:));%min distance
[row,column]=find(Dmatrix==sorted(1));
j=row(1);
i=column(1);
Dmatrix(i,:)=(Dmatrix(i,:)+Dmatrix(j,:))/2;%%average
Dmatrix(:,i)=(Dmatrix(:,i)+Dmatrix(:,j))/2;%%average
Dmatrix(i,i)=0;
r=find(Group(j,:)==1);
for k=1:1:size(r,2)
    Group(i,r(k))=1;
end
Group(j,j)=NaN;
Dmatrix(:,j)=NaN;
Dmatrix(j,:)=NaN;
B=diag(Dmatrix);%%Diagonal matrix
B=diag(B);%%Diagonal matrix
Dmatrix=Dmatrix-B;%%Diagonal matrix
out=Dmatrix;
[row column]=find(isnan(Group));
n=siz-size(row);
end
end
function [q res]=detecting(a,txt,txt2,nameOfway,file,th,num,cost,normal)
T=a.target(1,:);
Tn=a.nanTarget(1,:);
result=[T Tn];
result=sort(result);
word=zeros(1,2700);
n=0;
for i=1:1:size(result,2)
    tt=ismember(result(i),T);
    tn=ismember(result(i),Tn);
    if(tt==1)
        n=n+1;
      r=find(T==result(i));
      result(i)=1;  
       word(i)=a.target(2,r);
    else
      r=find(Tn==result(i));
      result(i)=num; 
      word(i)=a.nanTarget(2,r);
    end
end

result=result;
res=result;
data=epochTodata(txt2);
rowOftarget=find(result==1);

for i=1:1:length(rowOftarget)
    for j=1:1:10
    result=[result result(rowOftarget(i))];
    data=[data ;data(rowOftarget(i),:)];
    end
end
n1=size(data,1)
n2=size(data,2)
%SMOTE(data, result);
%choose algoritm
res=result;
if (nameOfway=='fitcsvm')

tabulate(result)
m1=fitcsvm(data,result,'Cost',cost);
else
m1=fitcdiscr(data,result,'Cost',cost);
end

data=epochTodata(txt);
if (file=='te')
word=a.Testtime(2,:);
end
[p1]=predict(m1,data);
p1=p1';
result=result';
rp=find(p1==num);
%rp2=find(p1==2);
%p1(rp)=[];
%result(rp)=[];
%testtarget=[result;p1];
word(rp)=[];

q=[];
for j=1:1:size(word,2)
    
           w=find(word==word(1,j));
           sz=size(w,2);
           if(sz>th-1 ||(normal==1&word(1,j)==12))
           q=[q,word(1,j)];
           if(word(1,j)==12)
           word(w(1,1:th-1))=NaN;
           else
           word(w(1,1:th))=NaN;
           end
           end

end
e=find(q==0);
q(e)=[];
end
function [n out]=WhichLetterTarget(txt)
a=load(txt);
data=a.train;
which=data(10,:);
turn=data(11,:);
collw=find(which~=0);
collt=find(turn~=0);
collfinal=collw-collw;
n=[];
q=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%target
target=which;
for j=1:1:size(collw,2)

       if(turn(1,collw(j))==0)
           target(1,collw(j))=NaN;
       end
       

end
for j=1:1:size(collw,2)
    
           w=find(target==target(1,collw(j)));
          
           sz=size(w,2);
           if(sz>4-1)
           n=[n,collw(j)];
           q=[q,target(1,collw(j))];
           target(w(1,1:4))=NaN;
           end
   

end
target=q;
out=q;
out2=0;


end
function [n out]=WhichLetterNanTarget(txt)
a=load(txt);
data=a.train;
which=data(10,:);
turn=data(11,:);
collw=find(which~=0);
collt=find(turn~=0);
collfinal=collw-collw;
n=[];
q=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% nan target
nontarget=which;
for j=1:1:size(collw,2)

       if(turn(1,collw(j))~=0)
           nontarget(1,collw(j))=NaN;
       end
       

end
for j=1:1:size(collw,2)
    
           w=find(nontarget==nontarget(1,collw(j)));
          
           sz=size(w,2);
           if(sz>4-1)
           n=[n,collw(j)];
           q=[q,nontarget(1,collw(j))];
           nontarget(w(1,1:4))=NaN;
           end
   

end
nontarget=q;
out=q;



end
function [n out]=WhichLetterTest(txt)
a=load(txt);
data=a.test;
which=data(10,:);
collw=find(which~=0);
collfinal=collw-collw;
n=[];
q=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%target
target=which;
for j=1:1:size(collw,2)
    
           w=find(target==target(1,collw(j)));
          
           sz=size(w,2);
           if(sz>4-1)
           n=[n,collw(j)];
           q=[q,target(1,collw(j))];
           target(w(1,1:4))=NaN;
           end
   

end
target=q;
out=q;
out2=0;


end
function [subject] =IndexExtraction(txt)
subject=load(txt);
[n out]=WhichLetterTarget(txt);
target=[n;out];
[n2 out2]=WhichLetterNanTarget(txt);
nanTarget=[n2;out2];
[n3 out3]=WhichLetterTest(txt);
Test=[n3;out3];
subject.target=target;
subject.nanTarget=nanTarget;
subject.Testtime=Test;
%save('MESubjectData2','a');

end
function [string]= detectLetter(txt , matrix)
string=[];
if (txt=='SC')
  num=matrix(1);
  switch num
      case 1
         string='A';
      case 2
         string='B';
      case 3
         string='C'; 
      case 4
         string='D';
      case 5
         string='E';
      case 6
         string='F';
      case 7
         string='G';
      case 8
         string='H';
      case 9
         string='I';
      case 10
         string='J';
      case 11
         string='K';
      case 12
         string='L';
      case 13
         string='M';
      case 14
         string='N';
      case 15
         string='O';
      case 16
         string='P';
      case 17
         string='Q';
      case 18
         string='R';
      case 19
         string='S';
      case 20
         string='T';
      case 21
         string='U';
      case 22
         string='V';
      case 23
         string='W';
      case 24
         string='X';
      case 25
         string='Y';
      case 26
         string='Z';
      case 27
         string='0';
      case 28
         string='1';
      case 29
         string='2';
      case 30
         string='3';
      case 31
         string='4';
      case 32
         string='5';
      case 33
         string='6';
      case 34
         string='7';
      case 35
         string='8';
      case 36
         string='9';
      otherwise
         string=[];
  end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   if (txt=='RC')
       matrix=sort(matrix);
       coll=matrix(1);
       row=matrix(2);
       if(row==7)
         switch coll
      case 1
         string='A';
      case 2
         string='B';
      case 3
         string='C'; 
      case 4
         string='D';
      case 5
         string='E';
      case 6
         string='F';
         
         end
       elseif (row==8)
        switch coll
      case 1
         string='G';
      case 2
         string='H';
      case 3
         string='I'; 
      case 4
         string='J';
      case 5
         string='K';
      case 6
         string='L';
         
         end
       elseif (row==9)
      switch coll
      case 1
         string='M';
      case 2
         string='N';
      case 3
         string='O'; 
      case 4
         string='P';
      case 5
         string='Q';
      case 6
         string='R';
         
         end
       elseif (row==10) 
     switch coll
      case 1
         string='S';
      case 2
         string='T';
      case 3
         string='U'; 
      case 4
         string='V';
      case 5
         string='W';
      case 6
         string='X';
         
         end
       elseif (row==11)
      switch coll
      case 1
         string='Y';
      case 2
         string='Z';
      case 3
         string='0'; 
      case 4
         string='1';
      case 5
         string='2';
      case 6
         string='3';
         
         end
       elseif (row==12)
      switch coll
      case 1
         string='4';
      case 2
         string='5';
      case 3
         string='6'; 
      case 4
         string='7';
      case 5
         string='8';
      case 6
         string='9';
         
         end
       end
   end

end
function data=epochTodata(txt)
epoch=load(txt);
cell = struct2cell(epoch);
epoch = double(cell2mat(cell));
n1=size(epoch,1);
n2=size(epoch,2);
n3=size(epoch,3);
for i=1:1:n2
    for j=1:1:n1
       data(i,n3*j-n3+1:n3*j)=epoch(j,i,:);
    end
end
end
function out=matrixOfletter(data,x)
target=data(2,:);
coll=find(target~=0);
out=[];
for i=1:1:length(coll)
    out=[out x(2,coll(i))];
end
end
function word=detectSC(q)
word=[];
for i=1:1:size(q,2)
    w=detectLetter('SC',q(i));
    word=[word w];  
end
end
function word=detectRC(q)
word=[];

for i=1:2:size(q,2)
    w=detectLetter('RC',[q(i) q(i+1)] );
    word=[word w];  
end
end
function [final_features final_mark] = SMOTE(original_features, original_mark)
ind = find(original_mark == 1);
% P = candidate points
P = original_features(ind,:);
T = P';
% X = Complete Feature Vector
X = T;
% Finding the 5 positive nearest neighbours of all the positive blobs
I = nearestneighbour(T, X, 'NumberOfNeighbours', 4);
I = I';
[r c] = size(I);
S = [];
th=0.3;
for i=1:r
    for j=2:c
        index = I(i,j);
        new_P=(1-th).*P(i,:) + th.*P(index,:);
        S = [S;new_P];
    end
end
original_features = [original_features;S];
[r c] = size(S);
mark = ones(r,1);
original_mark = [original_mark;mark];
train_incl = ones(length(original_mark), 1);
I = nearestneighbour(original_features', original_features', 'NumberOfNeighbours', 4);
I = I';
for j = 1:length(original_mark)
    len = length(find(original_mark(I(j, 2:4)) ~= original_mark(j,1)));
    if(len >= 2)
        if(original_mark(j,1) == 1)
         train_incl(original_mark(I(j, 2:4)) ~= original_mark(j,1),1) = 0;
        else
         train_incl(j,1) = 0;   
        end    
    end
end
final_features = original_features(train_incl == 1, :);
final_mark = original_mark(train_incl == 1, :);

end
function [idx, tri] = nearestneighbour(varargin)

error(nargchk(1, Inf, nargin, 'struct'));
% Default parameters
userParams.NumberOfNeighbours = []    ; % Finds one
userParams.DelaunayMode       = 'auto'; % {'on', 'off', |'auto'|}
userParams.Triangulation      = []    ;
userParams.Radius             = inf   ;
% Parse inputs
[P, X, fIndexed, userParams] = parseinputs(userParams, varargin{:});
% Special case uses Delaunay triangulation for speed.
% Determine whether to use Delaunay - set fDelaunay true or false
nX  = size(X, 2);
nP  = size(P, 2);
dim = size(X, 1);
switch lower(userParams.DelaunayMode)
    case 'on'
        %TODO Delaunay can't currently be used for finding more than one
        %neighbour
        fDelaunay = userParams.NumberOfNeighbours == 1 && ...
            size(X, 2) > size(X, 1)                    && ...
            ~fIndexed                                  && ...
            userParams.Radius == inf;
    case 'off'
        fDelaunay = false;
    case 'auto'
        fDelaunay = userParams.NumberOfNeighbours == 1 && ...
            ~fIndexed                                  && ...
            size(X, 2) > size(X, 1)                    && ...
            userParams.Radius == inf                   && ...
            ( ~isempty(userParams.Triangulation) || delaunaytest(nX, nP, dim) );
end
% Try doing Delaunay, if fDelaunay.
fDone = false;
if fDelaunay
    tri = userParams.Triangulation;
    if isempty(tri)
        try
            tri   = delaunayn(X');
        catch
            msgId = 'NearestNeighbour:DelaunayFail';
            msg = ['Unable to compute delaunay triangulation, not using it. ',...
                'Set the DelaunayMode parameter to ''off'''];
            warning(msgId, msg);
        end
    end
    if ~isempty(tri)
        try
            idx = dsearchn(X', tri, P')';
            fDone = true;
        catch
            warning('NearestNeighbour:DSearchFail', ...
                'dsearchn failed on triangulation, not using Delaunay');
        end
    end
else % if fDelaunay
    tri = [];
end
% If it didn't use Delaunay triangulation, find the neighbours directly by
% finding minimum distances
if ~fDone
    idx = zeros(userParams.NumberOfNeighbours, size(P, 2));
    % Loop through the set of points P, finding the neighbours
    Y = zeros(size(X));
    for iPoint = 1:size(P, 2)
        x = P(:, iPoint);
        % This is the faster than using repmat based techniques such as
        % Y = X - repmat(x, 1, size(X, 2))
        for i = 1:size(Y, 1)
            Y(i, :) = X(i, :) - x(i);
        end
        % Find the closest points, and remove matches beneath a radius
        dSq = sum(abs(Y).^2, 1);
        iRad = find(dSq < userParams.Radius^2);
        if ~fIndexed
            iSorted = iRad(minn(dSq(iRad), userParams.NumberOfNeighbours));
        else
            iSorted = iRad(minn(dSq(iRad), userParams.NumberOfNeighbours + 1));
            iSorted = iSorted(2:end);
        end
        % Remove any bad ones
        idx(1:length(iSorted), iPoint) = iSorted';
    end
    %while ~isempty(idx) && isequal(idx(end, :), zeros(1, size(idx, 2)))
    %    idx(end, :) = [];
    %end
    idx( all(idx == 0, 2), :) = [];
end % if ~fDone
if isvector(idx)
    idx = idx(:)';
end
end % nearestneighbour
function tf = delaunaytest(nx, np, dim)
switch dim
    case 2
        tf = np > min(1.5 * nx, 400);
    case 3
        tf = np > min(4 * nx  , 1200);
    case 4
        tf = np > min(40 * nx , 5000);
        % if the dimension is higher than 4, it is almost invariably better not
        % to try to use the Delaunay triangulation
    otherwise
        tf = false;
end % switch
end % delaunaytest
function I = minn(x, n)
% Make sure n is no larger than length(x)
n = min(n, length(x));
% Sort the first n
[xsn, I] = sort(x(1:n));
% Go through the rest of the entries, and insert them into the sorted block
% if they are negative enough
for i = (n+1):length(x)
    j = n;
    while j > 0 && x(i) < xsn(j)
        j = j - 1;
    end
    if j < n
        % x(i) should go into the (j+1) position
        xsn = [xsn(1:j), x(i), xsn((j+1):(n-1))];
        I   = [I(1:j), i, I((j+1):(n-1))];
    end
end
end %main
function [P, X, fIndexed, userParams] = parseinputs(userParams, varargin)
if length(varargin) == 1 || ~isnumeric(varargin{2})
    P           = varargin{1};
    X           = varargin{1};
    fIndexed    = true;
    varargin(1) = [];
else
    P             = varargin{1};
    X             = varargin{2};
    varargin(1:2) = [];
    % Check the dimensions of X and P
    if size(X, 1) ~= 1
        % Check to see whether P is in fact a vector of indices
        if size(P, 1) == 1
            try
                P = X(:, P);
            catch
                error('NearestNeighbour:InvalidIndexVector', ...
                    'Unable to index matrix using index vector');
            end
            fIndexed = true;
        else
            fIndexed = false;
        end % if size(P, 1) == 1
    else % if size(X, 1) ~= 1
        fIndexed = false;
    end
    if ~fIndexed && size(P, 1) ~= size(X, 1)
        error('NearestNeighbour:DimensionMismatch', ...
            'No. of rows of input arrays doesn''t match');
    end
end
% Parse the Property/Value pairs
if rem(length(varargin), 2) ~= 0
    error('NearestNeighbour:propertyValueNotPair', ...
        'Additional arguments must take the form of Property/Value pairs');
end
propertyNames = {'numberofneighbours', 'delaunaymode', 'triangulation', ...
    'radius'};
while length(varargin) ~= 0
    property = varargin{1};
    value    = varargin{2};
    % If the property has been supplied in a shortened form, lengthen it
    iProperty = find(strncmpi(property, propertyNames, length(property)));
    if isempty(iProperty)
        error('NearestNeighbour:InvalidProperty', 'Invalid Property');
    elseif length(iProperty) > 1
        error('NearestNeighbour:AmbiguousProperty', ...
            'Supplied shortened property name is ambiguous');
    end
    property = propertyNames{iProperty};
    switch property
        case 'numberofneighbours'
            if rem(value, 1) ~= 0 || ...
                    value > length(X) - double(fIndexed) || ...
                    value < 1
                error('NearestNeighbour:InvalidNumberOfNeighbours', ...
                    'Number of Neighbours must be an integer, and smaller than the no. of points in X');
            end
            userParams.NumberOfNeighbours = value;
        case 'delaunaymode'
            fOn = strcmpi(value, 'on');
            if strcmpi(value, 'off')
                userParams.DelaunayMode = 'off';
            elseif fOn || strcmpi(value, 'auto')
                if userParams.NumberOfNeighbours ~= 1
                    if fOn
                        warning('NearestNeighbour:TooMuchForDelaunay', ...
                            'Delaunay Triangulation method works only for one neighbour');
                    end
                    userParams.DelaunayMode = 'off';
                elseif size(X, 2) < size(X, 1) + 1
                    if fOn
                        warning('NearestNeighbour:TooFewDelaunayPoints', ...
                            'Insufficient points to compute Delaunay triangulation');
                    end
                    userParams.DelaunayMode = 'off';
                elseif size(X, 1) == 1
                    if fOn
                        warning('NearestNeighbour:DelaunayDimensionOne', ...
                            'Cannot compute Delaunay triangulation for 1D input');
                    end
                    userParams.DelaunayMode = 'off';
                else
                    userParams.DelaunayMode = value;
                end
            else
                warning('NearestNeighbour:InvalidOption', ...
                    'Invalid Option');
            end % if strcmpi(value, 'off')
        case 'radius'
            if isscalar(value) && isnumeric(value) && isreal(value) && value > 0
                userParams.Radius = value;
                if isempty(userParams.NumberOfNeighbours)
                    userParams.NumberOfNeighbours = size(X, 2) - double(fIndexed);
                end
            else
                error('NearestNeighbour:InvalidRadius', ...
                    'Radius must be a positive real number');
            end
    
        case 'triangulation'
            if isnumeric(value) && size(value, 2) == size(X, 1) + 1 && ...
                    all(ismember(1:size(X, 2), value))
                userParams.Triangulation = value;
            else
                error('NearestNeighbour:InvalidTriangulation', ...
                    'Triangulation not a valid Delaunay Triangulation');
            end
    end % switch property
    varargin(1:2) = [];
end % while
if isempty(userParams.NumberOfNeighbours)
    userParams.NumberOfNeighbours = 1;
end
end %parseinputs


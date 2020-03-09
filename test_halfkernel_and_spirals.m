%% twospirals
clear;close all;clc
N = 1000;
degrees= 720; %twospirals 
std=0.1;
% degrees= 0;
% std=0;
start = 0;
noise =0;
% noise = 0.9;
mean=0;

data = twospirals(N, degrees, start, noise,mean,std);
% color= [255 150 150; 150 150 255];
RGB1= repmat([255 0 0]/256, length(data)/2,1);
RGB2=repmat([0 0 255]/256, length(data)/2,1);
% RGB3=repmat([0 255 0]/256, length(data)/3,1);
% RGB=[RGB1;RGB2;RGB3];
RGB=[RGB1;RGB2];
figure;
% scatter3(data(:,1),data(:,2),data(:,3),5,RGB) %Red, Blue, Green(no additive noise)
scatter(data(:,1),data(:,2), 5,RGB)
grid;
title('Ex1')
xlabel('X axis');
ylabel('Y axis');


% figure;
% N=1000;
% noise =0;
% std=0;
% data = twospirals(N, degrees, start, noise,mean,std);
% RGB1= repmat([255 0 0]/256, length(data)/2,1);
% RGB2=repmat([0 0 255]/256, length(data)/2,1);
% RGB=[RGB1;RGB2];
% scatter(data(:,1),data(:,2), 5,RGB)
% grid;
% title('half spirals')
% xlabel('X axis');
% ylabel('Y axis');


% er = DNN(data(1:2,:),data(3,:));

%% halfkernel
clear;close all;clc
N = 1000;
minx = 0;
r1 = 20;
r2 = 25;
noise = 0;
ratio = 0.6;
data= halfkernel(N, minx, r1, r2, noise, ratio);
color= [255 150 150; 150 150 255];
RGB1= repmat([255 0 0]/256, length(data)/2,1);
RGB2=repmat([0 0 255]/256, length(data)/2,1);
RGB=[RGB1;RGB2];
figure;
scatter(data(:,1),data(:,2), 5,RGB)


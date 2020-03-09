clear all;
load('data.mat')
data=data';
Input_num = size(data,1)- 1;
input= data(1:Input_num,:);
output=data(Input_num+1,:);
Ni = Input_num;
%%
NumberOfNeurons = 3;
prompt = 'What is the number of layers?';
NumberOfLayers  = inputdlg(prompt);
NumberOfLayers = str2double(cell2mat(NumberOfLayers));
% disp(NumberOfLayers);
% if isempty(NumberOfLayers)
%     NumberOfLayers = 1;
% end
for layers=1:NumberOfLayers         %I can either add layers one by one, calculate error, choose the smaller
    Ni = NumberOfNeurons;           %calculate error, choose the smaller
    NumberOfNeurons = round((4*(Ni^2) + 3)/(Ni^2 - 8));
    NumberOfNeurons_list(layers)=NumberOfNeurons;
end
% NumberOfNeurons_list=[3,2];
h = NumberOfNeurons_list;
X = data(1:Input_num,:);
T = data(Input_num+1,:);
%%
model = [Input_num,NumberOfNeurons_list,1];
[erv,er, error_va,s]=DNN(data,model,T,200);
plot(error_va(1:s))
% binary = 0;
% er2=DNN([N_in,b,1],binary);

%% 
net=feedforwardnet(h);
net.performFcn = 'crossentropy';
[net ,tr]= train(net,X,T);
outputs = net(X);
errors = outputs - T;
plottrainstate(tr)
plotperform(tr)

%performance on different cost functions
perf1 = perform(net,outputs,T);
perf2 = crossentropy(net,outputs,T);
perf3 = mse(net,outputs,T);
perf4=mae(net,outputs,T);
perf5=sae(net,outputs,T);
figure; hold on;

plottrainstate(tr)
plotperform(tr)
%%
[model,mse] = mlp(X,T,h);
% plot(mse); 
disp(['T = [' num2str(T) ']']);
Y = mlpPred(model,X);
disp(['Y = [' num2str(Y) ']']);

    
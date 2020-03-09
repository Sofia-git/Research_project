clear;close all;clc;
% mkdir('Results//'); %Directory for Storing Resultsload('data.mat');

%% Data prep - Need change based on Data
load('data.mat'); %column data, for training set and test set 
load('data_val.mat'); %column data, validation set noiseless at all times 
Input_num = size(data,2)- 1; %without bias
Output_num=2; %update for non-binary classes
Input = data(:,1:Input_num);
Target = data(:,Input_num+1);
Output = -1*ones(size(Target));
%validation test
Input_val = data_val(:,1:Input_num);
Target_val = data_val(:,Input_num+1);
Output_val = -1*ones(size(Target_val));

Output_CE = -1*ones(size(Target));
Ni = Input_num;

%% Parameters
iter_max = 3000;
alpha = 0.01;
landa=0.002
epoch=20;
zero=0;
zero_CE=0;
Stop=-1;
adam_alpha = 0.001;
adam_b1 = 0.9;
adam_b2 = 0.999;
epsilon = sqrt(eps);
learningRate_plus = 1.2;
learningRate_negative = 0.5;
deltas_min = 10^-6;
deltas_max = 50;
%% Neurons and Layers (Sheela and Deepa number of Hidden neurons)
NumberOfNeurons = Input_num+1; %bias in first layer is included
prompt = 'What is the number of layers?';
NumberOfLayers  = inputdlg(prompt);
NumberOfLayers = str2double(cell2mat(NumberOfLayers));

for layers=1:NumberOfLayers         %I can either add layers one by one, calculate error, choose the smaller
    Ni = NumberOfNeurons;           %calculate error, choose the smaller
    NumberOfNeurons = round((4*(Ni^2) + 3)/(Ni^2 - 8));
    NumberOfNeurons_list(layers)=NumberOfNeurons;
end
% NumberOfNeurons_list=[1 1];%[10]wrs case 
L_nodes = [Input_num NumberOfNeurons_list Output_num];
L = length(L_nodes);
%% Adding Bias on each layer
L_nodes(1:end-1) = L_nodes(1:end-1) + 1;
Input = [ones(length(Input(:,1)),1) Input]; %input
Input_val = [ones(length(Input_val(:,1)),1) Input_val];

%% Weight initialization-MSE
W = cell(1, L); DW = cell(1, L); fx=cell(1, L); X=cell(1,L); e=cell(1,L);
ResilientDeltas=DW;
for j = 1:length(W)-1
    W{j} = 2*rand(L_nodes(j),L_nodes(j+1)) - ones(L_nodes(j),L_nodes(j+1)); % 0.2*ones(L_nodes(j),L_nodes(j+1)) - rand(L_nodes(j),L_nodes(j+1)); %RowIndex: From Node Number, ColumnIndex: To Node Number
    W{j}(:,1) = 0;
    DW{j} = zeros(L_nodes(j), L_nodes(j+1));
    %     fx{j+1}=zeros(L_nodes(j+1),1); %activation function results matrices should be the same as W
    X{j}= zeros(1,L_nodes(j));
    e{j}= zeros(1,L_nodes(j));
    ResilientDeltas{j} = 0.9*ones(L_nodes(j), L_nodes(j+1));
end
W{end} = 2*rand(L_nodes(end),1) - ones(L_nodes(end),1);
X{end} = zeros(1,L_nodes(end));
e{end} = zeros(1,L_nodes(end));
OldDW_Resilient = DW;
%initialise moment estimates
m = DW;
v=DW;
%% Weight initialization -CE
WW = cell(1, L); DWW = cell(1, L); fx=cell(1, L); XX=cell(1,L); CE=cell(1,L);
for j = 1:length(WW)-1
    WW{j} = 2*rand(L_nodes(j),L_nodes(j+1)) - ones(L_nodes(j),L_nodes(j+1)); % 0.2*ones(L_nodes(j),L_nodes(j+1)) - rand(L_nodes(j),L_nodes(j+1)); %RowIndex: From Node Number, ColumnIndex: To Node Number
    WW{j}(:,1) = 0;
    DWW{j} = zeros(L_nodes(j), L_nodes(j+1));
    %     fx{j+1}=zeros(L_nodes(j+1),1); %activation function results matrices should be the same as W
    XX{j}= zeros(1,L_nodes(j));
    CE{j}= zeros(1,L_nodes(j));
end
WW{end} = 2*rand(L_nodes(end),1) - ones(L_nodes(end),1);
XX{end} = zeros(1,L_nodes(end));
CE{end} = zeros(1,L_nodes(end));
%initialise moment estimates
mm = DWW;
v=DWW;

%% Creating output to use for number of output nodes (here binary)
Target_binary = zeros(length(Target), Output_num);
for i=1:length(Target)
    if (Target(i) == 1)
        Target_binary(i,:) = [1 0];
    elseif (Target(i) == 0)
        Target_binary(i,:) = [0 1];
    end
end

%% Training
N=length(Input); M=round(N*0.6); V = round(N*0.2); T=round(N*0.2);
sample=randsample(N,M+T+V);
ValidationError = zeros(iter_max,1);
ValidationError_CE = zeros(iter_max,1);
bound=0.5;

time1   = clock;
for iter = 1: iter_max
    %training set
    for i = 1:M
        %forward propagation
        X{1} = Input(sample(i),:);
        for j = 2:L
            X{j} = X{j-1}*W{j-1};
            X{j} = activation_function(X{j},'sigmoid');
            if (j ~= L)
                X{j}(1) = 1; % adding bias to all layers except the last
            end
        end
        e{L} =  Target_binary(sample(i),:)- X{L};
        %backward propagation and update
        for k = L-1:-1:1
            DX = derivative_function(X{k+1}, 'sigmoid');
            for neuron=1:length(e{k}) %updating each neuron on each layer
                e{k}(neuron) =  sum(e{k+1}.*DX.*W{k}(neuron,:) );
            end
        end
        
        for k = L:-1:2
            DX = derivative_function(X{k}, 'sigmoid');
            DW{k-1} = DW{k-1} + X{k-1}'*(e{k}.*DX);
        end
        %%%%% CE
        %forward propagation
%         XX{1} = Input(sample(i),:);
%         for j = 2:L
%             XX{j} = XX{j-1}*WW{j-1};
%             XX{j} = softmax(XX{j});
%             if (j ~= L)
%                 XX{j}(1) = 1; % adding bias to all layers except the last
%             end
%         end
%         %CE{L} =  (-Target_binary(sample(i),:)./XX{L})+((1-Target_binary(sample(i),:))./(1-XX{L}));
%         CE{L} =  Target_binary(sample(i),:)- X{L};
%         %backward propagation and update
%         for k = L-1:-1:1
%             DX = derivative_function(XX{k+1}, 'sigmoid');
%             for neuron=1:length(CE{k}) %updating each neuron on each layer
%                 CE{k}(neuron) =  sum(CE{k+1}.*DX.*WW{k}(neuron,:) );
%             end
%         end
%         
%         for k = L:-1:2
%             DX = derivative_function(XX{k}, 'sigmoid');
%              DWW{k-1} = DWW{k-1} + XX{k-1}'*(CE{k}.*DX);
%         end
    end
    if (mod(iter,200)==0) %Reset Deltas
        for Layer = 1:L
            ResilientDeltas{Layer} = alpha*DW{Layer};
        end
    end
        for Layer = 1:L-1
            mult = OldDW_Resilient{Layer} .* DW{Layer};
            ResilientDeltas{Layer}(mult > 0) = ResilientDeltas{Layer}(mult > 0) * learningRate_plus; % Sign didn't change
            ResilientDeltas{Layer}(mult < 0) = ResilientDeltas{Layer}(mult < 0) * learningRate_negative; % Sign changed
            ResilientDeltas{Layer} = max(deltas_min, ResilientDeltas{Layer});
            ResilientDeltas{Layer} = min(deltas_max, ResilientDeltas{Layer});

            OldDW_Resilient{Layer} = DW{Layer};

            DW{Layer} = sign(DW{Layer}) .* ResilientDeltas{Layer};
        end
        
    for Layer = 1:L
        DW{Layer} = alpha*DW{Layer} + 0.05*m{Layer};
%         DWW{Layer} = alpha*DWW{Layer} + 0.05*mm{Layer};
    end
    %     for i = 1:L
    %         m{i} = adam_b1*m{i} + (1 - adam_b1)* DW{i};    %update biased 1st moment estimate
    %         v{i} = adam_b2.*v{i} + (1 - adam_b1) .* (DW{i}.^2);  %update biased 2nd raw moment estimate
    %         mHat{i} = m{i}./(1 - adam_b1^i);  %Compute bias-corrected 1st moment estimate
    %         vHat{i} = v{i}./(1 - adam_b2^i);  %Compute bias-corrected 2nd raw moment estimate
    %         DW{i} = -adam_alpha.*mHat{i}./(sqrt(vHat{i}) + epsilon); % - Determine step to take at this iteration
    %     end
    %weight update
    for i = 1:L-1
        W{i} = W{i} + DW{i};
%         WW{i} = WW{i} + DWW{i};
    end
    for Layer = 1:length(DW)
        DW{Layer} = 0 * DW{Layer};
%         DWW{Layer} = 0 * DWW{Layer};
    end
    %validation
    for s=M+1:M+V
        %forward propagation
%         X{1} = Input_val(sample(s),:);
%         for j = 2:L
%             X{j} = X{j-1}*W{j-1};
%             X{j} = activation_function(X{j},'sigmoid');
%             if (j ~= L)
%                 X{j}(1) = 1; % adding bias to all layers except the last
%             end
%         end
%         result = X{end};
%         
%         if (result(1) >= bound && result(2) < bound) %TODO: Not generic role for any number of output nodes
%             Output_val(sample(s)) = 1;
%         elseif (result(1) < bound && result(2) >= bound)
%             Output_val(sample(s)) = 0;
%         else
%             if (result(1) >= result(2))
%                 Output_val(sample(s)) = 1;
%             else
%                 Output_val(sample(s)) = 0;
%             end
%         end
        X{1} = Input(sample(s),:);
        for j = 2:L
            X{j} = X{j-1}*W{j-1};
            X{j} = activation_function(X{j},'sigmoid');
            if (j ~= L)
                X{j}(1) = 1; % adding bias to all layers except the last
            end
        end
        result = X{end};
        
        if (result(1) >= bound && result(2) < bound) %TODO: Not generic role for any number of output nodes
            Output(sample(s)) = 1;
        elseif (result(1) < bound && result(2) >= bound)
            Output(sample(s)) = 0;
        else
            if (result(1) >= result(2))
                Output(sample(s)) = 1;
            else
                Output(sample(s)) = 0;
            end
        end
        %forward propagation
%         XX{1} = Input(sample(s),:);
%         for j = 2:L
%             XX{j} = XX{j-1}*WW{j-1};
%             XX{j} = softmax(XX{j});
%             if (j ~= L)
%                 XX{j}(1) = 1; % adding bias to all layers except the last
%             end
%         end
%         result_CE = XX{end};
%         
%         if (result_CE(1) >= bound && result_CE(2) < bound) %TODO: Not generic role for any number of output nodes
%             Output_CE(sample(s))= Output_CE(sample(s)) > 0.5; % 1-(exp(-12));
%         elseif (result_CE(1) < bound && result_CE(2) >= bound)
%             Output_CE(sample(s))= Output_CE(sample(s)) < 0.5;% -(exp(-12));
%         else
%             if (result_CE(1) >= result_CE(2))
%                 Output_CE(sample(s))= Output_CE(sample(s)) > 0.5;%-(exp(-12));
%             else
%                 Output_CE(sample(s))=Output_CE(sample(s)) < 0.5;%-(exp(-12));
%             end
%         end
     end
%     ValidationError(iter) = ErrorFunction(Output_val(sample(M+1:M+V)),Target(sample(M+1:M+V)),'MSE')/(V-1); %Average Validation Error
%     if (ValidationError(iter) == 0)
%         zero = 1;
%     end
%one layer
% [b]=X{1:end};
% overall= (sum(b)-1)^2;
% %two layer
% [b,c]=X{1:end};
% overall= (sum(b)+sum(c)-2)^2;
%three layer
% [b,c,d]=X{1:end};
% overall= (sum(b)+sum(c)+sum(d)-3)^2;
overall=0;
    ValidationError(iter) = (ErrorFunction(Output(sample(M+1:M+V)),Target(sample(M+1:M+V)),'MSE') + (landa*overall) )/(V-1); %Average Validation Error
%     - X{1:end-1}(1)
    if (ValidationError(iter) <= 0.0001)
        zero = 1;
    end
%     ValidationError_CE(iter) = ErrorFunction(Output_CE(sample(M+1:M+V)),Target(sample(M+1:M+V)),'CE')/(V-1); %Average Validation Error
%     if (ValidationError_CE(iter) ==0) %-(exp(-12)))
%         zero_CE = 1;
%     end
    %% Testing
    for s=M+V+1:M+T+V
        %forward propagation
        X{1} = Input(sample(s),:);
        for j = 2:L
            X{j} = X{j-1}*W{j-1};
            X{j} = activation_function(X{j},'sigmoid');
            if (j ~= L)
                X{j}(1) = 1; % adding bias to all layers except the last
            end
        end
        result = X{end};
        if (result(1) >= result(2))
            Output(sample(s)) = 1;
        else
            Output(sample(s)) = 0;
        end
        TestError(iter) = ErrorFunction(Output(sample(M+1:M+V)),Target(sample(M+1:M+V)),'MSE')/(M+1); %Average Validation Error
        %forward propagation
%         XX{1} = Input(sample(s),:);
%         for j = 2:L
%             XX{j} = XX{j-1}*WW{j-1};
%             XX{j} = activation_function(XX{j},'sigmoid');
%             if (j ~= L)
%                 XX{j}(1) = 1; % adding bias to all layers except the last
%             end
%         end
%         result_CE = XX{end};
%         if (result_CE(1) >= result_CE(2))
%             Output_CE(sample(s)) = 1;
%         else
%             Output_CE(sample(s)) = 0;
%         end
%         TestError_CE(iter) = ErrorFunction(Output_CE(sample(M+1:M+V)),Target(sample(M+1:M+V)),'CE')/(M+1); %Average Validation Error
     end
 time2   = clock;
 fprintf('CLOCK:   %g\n', etime(time2, time1));
    %% Plots
    if (zero || zero_CE || mod(iter,epoch)==0)
        %Decision Boundary
        unique_TargetClasses = unique(Target);
        training_colors = {'y.', 'b.'};
        separation_colors = {'g.', 'r.'};
%         subplot(2,2,1);
        subplot(2,1,1);
        cla;
        hold on;
        title(['Decision Boundary at Epoch Number ' int2str(iter) '.']);
        
        margin = 0.05; step = 0.05;
        xlim([min(Input(:,2))-margin max(Input(:,2))+margin]);
        ylim([min(Input(:,3))-margin max(Input(:,3))+margin]);
        for x = min(Input(:,2))-margin : step : max(Input(:,2))+margin
            for y = min(Input(:,3))-margin : step : max(Input(:,3))+margin
                X{1} = [1 x y];
                for j = 2:L
                    X{j} = X{j-1}*W{j-1};
                    X{j} = activation_function(X{j},'sigmoid');
                    if (j ~= L)
                        X{j}(1) = 1; % adding bias to all layers except the last
                    end
                end
                result=X{end};
                bound = 0.5;
                if (result(1) >= bound && result(2) < bound) %TODO: Not generic role for any number of output nodes
                    plot(x, y, separation_colors{1}, 'markersize', 18);
                elseif (result(1) < bound && result(2) >= bound)
                    plot(x, y, separation_colors{2}, 'markersize', 18);
                else
                    if (result(1) >= result(2))
                        plot(x, y, separation_colors{1}, 'markersize', 18);
                    else
                        plot(x, y, separation_colors{2}, 'markersize', 18);
                    end
                end
            end
            
        end
        for i = 1:length(unique_TargetClasses)
            points = Input(Target==unique_TargetClasses(i), 2:end);
            plot(points(:,1), points(:,2), training_colors{i}, 'markersize', 10);
        end
        axis equal;
        
%         unique_TargetClasses = unique(Target);
%         training_colors = {'y.', 'b.'};
%         separation_colors = {'g.', 'r.'};
%         subplot(2,2,2);
%         cla;
%         hold on;
%         title(['Decision Boundary at Epoch Number ' int2str(iter) '.']);
%         
%         margin = 0.05; step = 0.05;
%         xlim([min(Input(:,2))-margin max(Input(:,2))+margin]);
%         ylim([min(Input(:,3))-margin max(Input(:,3))+margin]);
%         for x = min(Input(:,2))-margin : step : max(Input(:,2))+margin
%             for y = min(Input(:,3))-margin : step : max(Input(:,3))+margin
%                 XX{1} = [1 x y];
%                 for j = 2:L
%                     XX{j} = XX{j-1}*WW{j-1};
%                     XX{j} = activation_function(XX{j},'sigmoid');
%                     if (j ~= L)
%                         XX{j}(1) = 1; % adding bias to all layers except the last
%                     end
%                 end
%                 
%                 result_CE=XX{end};
%                 bound = 0.5;
%                 if (result_CE(1) >= bound && result_CE(2) < bound) %TODO: Not generic role for any number of output nodes
%                     plot(x, y, separation_colors{1}, 'markersize', 18);
%                 elseif (result_CE(1) < bound && result_CE(2) >= bound)
%                     plot(x, y, separation_colors{2}, 'markersize', 18);
%                 else
%                     if (result_CE(1) >= result_CE(2))
%                         plot(x, y, separation_colors{1}, 'markersize', 18);
%                     else
%                         plot(x, y, separation_colors{2}, 'markersize', 18);
%                     end
%                 end
%             end
%         end
%         
%         for i = 1:length(unique_TargetClasses)
%             points = Input(Target==unique_TargetClasses(i), 2:end);
%             plot(points(:,1), points(:,2), training_colors{i}, 'markersize', 10);
%         end
%         axis equal;
        
        % Draw Mean Square Error
%         subplot(2,2,3);
        subplot(2,1,2);
        ValidationError(ValidationError==-1) = [];
        plot([ValidationError(1:iter)]);
        ylim([-0.1 0.6]);
        title('Error Function');
        xlabel('Epochs');
        ylabel('MSE');
        grid on;
        %         hold on;
%         subplot(2,2,4);
%         ValidationError_CE(ValidationError_CE==-1) = [];
%         plot([ValidationError_CE(1:iter)]);
%         %         ylim([-0.1 0.6]);
%         title('Error Function');
%         xlabel('Epochs');
%         ylabel('MSE & CE');
%         grid on;
        %
        saveas(gcf, sprintf('Results//fig%i.eps', iter),'eps');
%         pause(0.05);
        %         hold on
    end
    display([int2str(iter) ' Epochs. MSE = ' num2str(ValidationError(iter))'.']);
%     display([int2str(iter) ' Epochs. CE = ' num2str(ValidationError_CE(iter))'.']);

    if (zero)
        break;
    end
end
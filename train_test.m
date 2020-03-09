%% Weight initialization
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
ValidationError_CE = zeros(iter_max,1);
bound=0.5;
  
for iter = 1: iter_max
    %training set
    for i = 1:M
        %forward propagation
        XX{1} = Input(sample(i),:);
        for j = 2:L
            XX{j} = XX{j-1}*WW{j-1};
            XX{j} = activation_function(XX{j},'sigmoid');
            if (j ~= L)
                XX{j}(1) = 1; % adding bias to all layers except the last
            end
        end
        CE{L} =  (-Target_binary(sample(i),:)/XX{L})+((1-Target_binary(sample(i),:))/(1-XX{L}));
        %backward propagation and update
        for k = L-1:-1:1
            DX = derivative_function(XX{k+1}, 'sigmoid');
            for neuron=1:length(CE{k}) %updating each neuron on each layer
                CE{k}(neuron) =  sum(CE{k+1}.*DX.*WW{k}(neuron,:) );
            end
        end
        
        for k = L:-1:2
            DX = derivative_function(XX{k}, 'sigmoid');
            DWW{k-1} = DWW{k-1} + XX{k-1}'*(CE{k}.*DX);
        end
         %%%%%
    end
    for Layer = 1:L
        DWW{Layer} = alpha*DWW{Layer} + 0.05*mm{Layer};
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
        WW{i} = WW{i} + DWW{i};
    end
    for Layer = 1:length(DWW)
        DWW{Layer} = 0 * DWW{Layer};
    end
    %validation
    for s=M+1:M+V
        %forward propagation
        XX{1} = Input(sample(s),:);
        for j = 2:L
            XX{j} = XX{j-1}*WW{j-1};
            XX{j} = activation_function(XX{j},'sigmoid');
            if (j ~= L)
                XX{j}(1) = 1; % adding bias to all layers except the last
            end
        end
        result_CE = XX{end};
        
        if (result_CE(1) >= bound && result_CE(2) < bound) %TODO: Not generic role for any number of output nodes
            Output_CE(sample(s)) = 1;
        elseif (result_CE(1) < bound && result_CE(2) >= bound)
            Output_CE(sample(s)) = 0;
        else
            if (result_CE(1) >= result_CE(2))
                Output_CE(sample(s)) = 1;
            else
                Output_CE(sample(s)) = 0;
            end
        end
    end
    ValidationError_CE(iter) = ErrorFunction(Output_CE(sample(M+1:M+V)),Target(sample(M+1:M+V)),'MSE')/(V-1); %Average Validation Error
    if (ValidationError_CE(iter) == 0)
        zero = 1;
    end
    %% Testing
    for s=M+V+1:M+T+V
        %forward propagation
        XX{1} = Input(sample(s),:);
        for j = 2:L
            XX{j} = XX{j-1}*WW{j-1};
            XX{j} = activation_function(XX{j},'sigmoid');
            if (j ~= L)
                XX{j}(1) = 1; % adding bias to all layers except the last
            end
        end
        result_CE = XX{end};
        if (result_CE(1) >= result_CE(2))
            Output_CE(sample(s)) = 1;
        else
            Output_CE(sample(s)) = 0;
        end
        TestError_CE(iter) = ErrorFunction(Output_CE(sample(M+1:M+V)),Target(sample(M+1:M+V)),'MSE')/(M+1); %Average Validation Error
    end
    
    %% Plots
    if (zero || mod(iter,epoch)==0)
        %Decision Boundary
        unique_TargetClasses = unique(Target);
        training_colors = {'y.', 'b.'};
        separation_colors = {'g.', 'r.'};
        subplot(2,1,1);
        cla;
        hold on;
        title(['Decision Boundary at Epoch Number ' int2str(iter) '.']);
        
        margin = 0.05; step = 0.05;
        xlim([min(Input(:,2))-margin max(Input(:,2))+margin]);
        ylim([min(Input(:,3))-margin max(Input(:,3))+margin]);
        for x = min(Input(:,2))-margin : step : max(Input(:,2))+margin
            for y = min(Input(:,3))-margin : step : max(Input(:,3))+margin
                XX{1} = [1 x y];
                for j = 2:L
                    XX{j} = XX{j-1}*WW{j-1};
                    XX{j} = activation_function(XX{j},'sigmoid');
                    if (j ~= L)
                        XX{j}(1) = 1; % adding bias to all layers except the last
                    end
                end
                result_CE=XX{end};
                bound = 0.5;
                if (result_CE(1) >= bound && result_CE(2) < bound) %TODO: Not generic role for any number of output nodes
                    plot(x, y, separation_colors{1}, 'markersize', 18);
                elseif (result_CE(1) < bound && result_CE(2) >= bound)
                    plot(x, y, separation_colors{2}, 'markersize', 18);
                else
                    if (result_CE(1) >= result_CE(2))
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
        
        % Draw Mean Square Error
        subplot(2,1,2);
        ValidationError_CE(ValidationError_CE==-1) = [];
        plot([ValidationError_CE(1:iter)]);
        ylim([-0.1 0.6]);
        title('Error Function');
        xlabel('Epochs');
        ylabel('CE');
        grid on;
%         
        saveas(gcf, sprintf('Results//fig%i.png', iter),'jpg');
        pause(0.05);
%         hold on
    end
    display([int2str(iter) ' Epochs. CE = ' num2str(ValidationError_CE(iter))'.']);
    if (zero)
        break;
    end
end
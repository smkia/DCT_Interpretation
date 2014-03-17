%% Simulations
clear all;
load('2DLayout.mat'); % loading 2D-layout
foldNum = 5;
alpha = 1;
phaseVar = [1,45,90,135,180,225,270,315]; % Phase variability of background noise.
chNum = 8:-1:1;  % Number of targets channels
t = -0.5:0.01:0.49;
channels = [59, 60, 61, 62, 63, 64, 65, 66];
simulationNum = 100;
for cLoop = 1 : length(chNum)      % This loops provides the variability for number of target channels. (n_tc in the paper)
    for pLoop = 1 : length(phaseVar) % This loops provides different data complexity based on variability of phase of background activity. Not mentioned in the paper but the results are used.
        for z = 1 : simulationNum    % The simulation loop
            % data simulation
            data.trial = [];
            data.time = [];
            targetPattern = zeros(100,102,100);
            for i = 1 : 200         % Trial simulation, 100 class A with target effect, 100 class B without target effect
                c = randperm(length(channels),chNum(cLoop)); % selecting cLoop channels randomly from 59-66 channles. 
                affectedChannels = channels(c);
                phi = randi(phaseVar(pLoop));       % selecting random phase for background noise
                phi2 = deg2rad(randi(180));         % selecting random phase for target effect
                for j = 1 : 102             % Loop for all channels
                    temp = [];
                    if sum(j == affectedChannels) == 0 % if the channel is not in the target channel set
                        temp = 2*sin(2*pi*5*t+phi2); % Add the background noise
                    elseif i <= 100                 % if the channels is in target channel set and just for target trials in Class A (first 100)
                        temp = chirp(t,7,0.5,15,'quadratic',phi,'convex')+sin(2*pi*5*t+phi2); % Add the chirp effect
                        targetPattern(i,j,:) = squeeze(targetPattern(i,j,:)) + temp';
                    else
                        temp = 2*sin(2*pi*5*t+phi2);    % Add just background noise to the target channels of trials in Class B (second 100)
                    end
                    data.trial{i}(j,:) = randn(1,100)+temp(1:100); % add White noise to all channels and trials
                    data.time{i} = 0:0.01:0.99;
                end
            end
            
            % redundant stuff that help this script work, will be removed
            % soon
            for i = 1 : length (data.trial)
                trials(i,:,:) = data.trial{i};
            end
            [trialNum,channelNum,timeNum] = size(trials);
            for i = 1 : 100
                tp(:,i) = reshape(squeeze(targetPattern(i,:,:))',102*100,1);
            end
            targetPattern = tp;
            
            % Defining target vector
            targets = ones(1,200);
            targets(101:200) = -1;
            
            % partitioning the data for cross validation
            CVO = cvpartition(targets,'k',foldNum);
            
            % Here on we are going to train SLR on whole features in original space
            for i = 1 : trialNum
                features(:,i) = reshape(squeeze(trials(i,:,:))',102*100,1); % features are accumulated channels
            end
            features = mapstd(features);   % Standardizing features
            % Parameter Estimation
            options = statset('UseParallel','always');
            options.UseParallel = true;
            if z == 1 % Estimating lambda value for regularization. To speed up, this part is executing in first run of simulation for each 
                %parameter and the labda will be used for similar simulations.
                lambda = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1];
                [~,fitInfo] = lasso(features(1:end,:)',targets','CV',5,'Lambda',lambda,'Options',options,'Alpha',alpha,'Standardize',false);
                lambdaWhole = fitInfo.LambdaMinMSE;
            end
            % Here is the cross validation to estimate the accuracy of the
            % model
            for i = 1:CVO.NumTestSets
                trIdx = CVO.training(i);
                teIdx = CVO.test(i);
                [B{i},Intercept{i}] = lassoFit1(features(:,trIdx)',targets(trIdx)',[],lambdaWhole,alpha,Inf,0,1e-4,1000,ones(1,size(features,1)),1,0);
                covMat{i} = cov(features(:,trIdx)');
                trainingOutputs = [ones(size(features(:,trIdx)',1),1) features(:,trIdx)'] * [Intercept{i};B{i}];
                testOutputs = [ones(size(features(:,teIdx)',1),1) features(:,teIdx)'] * [Intercept{i};B{i}];
                [~,~,~,tsAcc(i)] = perfcurve(targets(teIdx)',testOutputs,1);
                [~,~,~,trAcc(i)] = perfcurve(targets(trIdx)',trainingOutputs,1);
                disp(strcat(num2str(cLoop),':',num2str(pLoop),':',num2str(z),':',num2str(i),':',num2str(tsAcc(i))));
            end
            
            % redundant stuff that help this script work, will be removed
            % soon
            for i = 1 : CVO.NumTestSets
                for j = 1 : 100
                    c(i,j) = corr(targetPattern(:,j),B{i});
                end
            end
            patternCorrWhole(cLoop,pLoop,z) = mean2(c);
            v = var(cell2mat(B),0,2);
            v = v(v~=0);
            patternVarWhole(cLoop,pLoop,z) = mean(v);
            
            % Computing Mean accuracy over k-folds
            tsACCWhole(cLoop,pLoop,z) = mean(tsAcc);
            trACCWhole(cLoop,pLoop,z) = mean(trAcc);
            
            % Computing the mean correlation between weight vectors on
            % different runs of cross validation. This is done on pure SLR
            % weights.
            tmp = tril(corr(cell2mat(B)),-1);
            tmp = tmp(tmp~=0);
            coefCorrWhole(cLoop,pLoop,z) = mean(tmp);
            
            % Here we compute covariance matrix of inputs and multiply it
            % by SLR weights. See [2] in the paper.
            for runNum = 1 : foldNum
                Bbar{runNum} = covMat{runNum}*B{runNum};
            end
            
            % redundant stuff that help this script work, will be removed
            % soon
            for i = 1 : CVO.NumTestSets
                for j = 1 : 100
                    c(i,j) = corr(targetPattern(:,j),Bbar{i});
                end
            end
            patternCorrCov(cLoop,pLoop,z) = mean2(c);
            v = var(cell2mat(Bbar),0,2);
            v = v(v~=0);
            patternVarCov(cLoop,pLoop,z) = mean(v);
            
            % Computing the mean correlation between corrected weight vectors on
            % different runs of cross validation. This is done on the
            % result of [2] Procedure.
            tmp = tril(corr(cell2mat(Bbar)),-1);
            tmp = tmp(tmp~=0);
            coefCorrCov(cLoop,pLoop,z) = mean(tmp);
            
            
            % Fom here on we transform the data to DCT space an repeat the
            % same stuff. See the paper.
            
            % Rearranging the sensors based on a 2D map
            wholeTrials2DGRAD = zeros(trialNum,11,10,timeNum);
            for i = 1 : trialNum
                for j = 1 : 204
                    if labelMap{j,1}(7) == '2'
                        wholeTrials2DGRAD(i,labelMap{j,2}(1),labelMap{j,2}(2),:) = trials(i,j,:);
                    end
                end
            end
            
            coefNum = [5,5,25]; % number of DCT coefficients to be used in the feature vector
            n = coefNum(1)*coefNum(2)*coefNum(3);
            featuresGrad = zeros(n,trialNum);
            % Applying 3D-DCT
            DGRAD = zeros(size(wholeTrials2DGRAD));
            for i = 1 : trialNum
                DGRAD(i,:,:,:) = shiftdim(dct3(squeeze(wholeTrials2DGRAD(i,:,:,:))),-1);
                featuresGrad(:,i) = reshape(squeeze(DGRAD(i,1:coefNum(1),1:coefNum(2),1:coefNum(3))),n,1);
            end
            [featuresGrad] = mapstd(featuresGrad); % Normalizing the features
            
            % Estimating the lamda for regularization
            options = statset('UseParallel','always');
            options.UseParallel = true;
            if z == 1
                lambda = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1];
                [~,fitInfo] = lasso(featuresGrad(1:end,:)',targets','CV',5,'Lambda',lambda,'Options',options,'Alpha',alpha,'Standardize',false);
                lambdaDCT = fitInfo.LambdaMinMSE;
            end
            % Cross-validation
            parfor i = 1:CVO.NumTestSets
                trIdx = CVO.training(i);
                teIdx = CVO.test(i);
                [BDCT{i},InterceptDCT{i}] = lassoFit1(featuresGrad(:,trIdx)',targets(trIdx)',[],lambdaDCT,alpha,Inf,0,1e-4,1000,ones(1,size(featuresGrad,1)),1,0);
                trainingOutputs = [ones(size(featuresGrad(:,trIdx)',1),1) featuresGrad(:,trIdx)'] * [InterceptDCT{i};BDCT{i}];
                testOutputs = [ones(size(featuresGrad(:,teIdx)',1),1) featuresGrad(:,teIdx)'] * [InterceptDCT{i};BDCT{i}];
                [~,~,~,tsAcc(i)] = perfcurve(targets(teIdx)',testOutputs,1);
                [~,~,~,trAcc(i)] = perfcurve(targets(trIdx)',trainingOutputs,1);
                disp(strcat(num2str(cLoop),':',num2str(pLoop),':',num2str(z),':',num2str(i),':',num2str(tsAcc(i))));
            end
            
            % Applying Inverse 3D-DCT to the weights of SLR and rearranging
            % them to the default arrangment of channels
            for runNum = 1 : foldNum
                D = zeros(11,10,100);
                temp = reshape(BDCT{runNum},coefNum(1),coefNum(2),coefNum(3));
                D(1:coefNum(1),1:coefNum(2),1:coefNum(3)) = temp;
                tempMap = idct3(D);
                l = 1;
                for j = 1 : 204
                    if labelMap{j,1}(7) == '1'
                        m{l,1} =  labelMap{j,2};
                        l = l+1;
                    end
                end
                for i = 1 : 102
                    map{runNum}(i,:) = tempMap(m{i,1}(1),m{i,1}(2),:);
                end
            end
            
            % Redundant stuff, will be removed soon
            for i = 1 : CVO.NumTestSets
                tmp1{i} = reshape(map{i}',102*100,1);
                for j = 1 : 100
                    c(i,j) = corr(targetPattern(:,j),tmp1{i});
                end
            end
            patternCorrDCT(cLoop,pLoop,z) = mean2(c);
            v = var(cell2mat(tmp1),0,2);
            v = v(v~=0);
            patternVarDCT(cLoop,pLoop,z) = mean(v);
            
            % Computing mean accuracy over folds
            tsACCDCT(cLoop,pLoop,z) = mean(tsAcc);
            trACCDCT(cLoop,pLoop,z) = mean(trAcc);
            % Computing mean correlation of corrected weights over
            % different folds
            tmp = tril(corr(cell2mat(BDCT)),-1);
            tmp = tmp(tmp~=0);
            coefCorrDCT(cLoop,pLoop,z) = mean(tmp);
            tmp = tril(corr(cell2mat(tmp1)),-1);
            tmp = tmp(tmp~=0);
            coefCorrDCT1(cLoop,pLoop,z) = mean(tmp);
            % saving the stuff
            save('tempFile','patternCorrDCT','patternVarDCT','tsACCDCT','trACCDCT','patternCorrWhole','patternVarWhole','tsACCWhole',...
                'trACCWhole','patternCorrCov','patternVarCov','coefCorrDCT','coefCorrDCT1','coefCorrCov','coefCorrWhole');
        end
    end
end

%% Summarizing and plotting the results
% IM
chanceLevel = 0.5;
p = 8;
D = poslin(tsACCDCT(:,1:p,:) - chanceLevel) .* poslin(coefCorrDCT1(:,1:p,:)) / (1 - chanceLevel);
C = poslin(tsACCWhole(:,1:p,:) - chanceLevel) .* poslin(coefCorrCov(:,1:p,:)) / (1 - chanceLevel);
W = poslin(tsACCWhole(:,1:p,:) - chanceLevel) .* poslin(coefCorrWhole(:,1:p,:)) / (1 - chanceLevel);
errorbar(mean(mean(D,3),2),std(reshape(D,8,800),[],2),'rd-','MarkerSize',10,'LineWidth',3);hold on;
errorbar(mean(mean(C,3),2),std(reshape(C,8,800),[],2),'bs-','MarkerSize',10,'LineWidth',3);hold on;
errorbar(mean(mean(W,3),2),std(reshape(W,8,800),[],2),'gv-','MarkerSize',10,'LineWidth',3);hold off;
set(gca,'XTickLabel',{'','8','7','6','5','4','3','2','1'},'fontSize',12,'fontWeight','bold','LineWidth',2);
title('');
ylabel('IM');
xlabel('n_{tc}');
ylim([0,1]);
legend('3D-DCT','Baseline','SLR');
% AM
p = 8;
D = tsACCDCT(:,1:p,:);
C = tsACCWhole(:,1:p,:);
W = tsACCWhole(:,1:p,:);
figure;
errorbar(mean(mean(D,3),2),std(reshape(D,8,800),[],2),'rd-','MarkerSize',10,'LineWidth',3);hold on;
errorbar(mean(mean(C,3),2),std(reshape(C,8,800),[],2),'bs-','MarkerSize',10,'LineWidth',3);hold on;
errorbar(mean(mean(W,3),2),std(reshape(W,8,800),[],2),'gv-','MarkerSize',10,'LineWidth',3);hold off;
set(gca,'XTickLabel',{'','8','7','6','5','4','3','2','1'},'fontSize',12,'fontWeight','bold','LineWidth',2);
title('');
ylabel('AM');
xlabel('n_{tc}');
ylim([0,1]);
legend('3D-DCT','Baseline','SLR');
% CM
p = 8;
D = poslin(coefCorrDCT1(:,1:p,:));
C = poslin(coefCorrCov(:,1:p,:));
W = poslin(coefCorrWhole(:,1:p,:));
figure;
errorbar(mean(mean(D,3),2),std(reshape(D,8,800),[],2),'rd-','MarkerSize',10,'LineWidth',3);hold on;
errorbar(mean(mean(C,3),2),std(reshape(C,8,800),[],2),'bs-','MarkerSize',10,'LineWidth',3);hold on;
errorbar(mean(mean(W,3),2),std(reshape(W,8,800),[],2),'gv-','MarkerSize',10,'LineWidth',3);hold off;
set(gca,'XTickLabel',{'','8','7','6','5','4','3','2','1'},'fontSize',12,'fontWeight','bold','LineWidth',2);
title('');
ylabel('CM');
xlabel('n_{tc}');
ylim([0,1]);
legend('3D-DCT','Baseline','SLR');

%% Simulations
clear all;
foldNum = 5;
alpha = 1;
phaseVar = [1,45,90,135,180,225,270,315]; % Phase variability of background noise.
chNum = 8:-1:1;  % Number of targets channels
t = -0.5:0.01:0.49;
channels = [59, 60, 61, 62, 63, 64, 65, 66];
simulationNum = 100;
for cLoop = 1 : length(chNum)
    for pLoop = 1 : length(phaseVar)
        for z = 1 : simulationNum
            % data simulation
            data.trial = [];
            data.time = [];
            targetPattern = zeros(100,102,100);
            for i = 1 : 200
                c = randperm(length(channels),chNum(cLoop));
                affectedChannels = channels(c);
                phi = randi(phaseVar(pLoop));
                phi2 = deg2rad(randi(180));
                for j = 1 : 102
                    temp = [];
                    if sum(j == affectedChannels) == 0
                        temp = 2*sin(2*pi*5*t+phi2);
                    elseif i <= 100
                        temp = chirp(t,7,0.5,15,'quadratic',phi,'convex')+sin(2*pi*5*t+phi2);
                        targetPattern(i,j,:) = squeeze(targetPattern(i,j,:)) + temp';
                    else
                        temp = 2*sin(2*pi*5*t+phi2);
                    end
                    data.trial{i}(j,:) = randn(1,100)+temp(1:100);
                    data.time{i} = 0:0.01:0.99;
                end
            end
            for i = 1 : length (data.trial)
                trials(i,:,:) = data.trial{i};
            end
            targets = ones(1,200);
            targets(101:200) = -1;
            [trialNum,channelNum,timeNum] = size(trials);
            for i = 1 : 100
                tp(:,i) = reshape(squeeze(targetPattern(i,:,:))',102*100,1);
            end
            targetPattern = tp;
            CVO = cvpartition(targets,'k',foldNum);
            
            % Whole Features
            for i = 1 : trialNum
                features(:,i) = reshape(squeeze(trials(i,:,:))',102*100,1);
            end
            features = mapstd(features);
            % Parameter Estimation
            options = statset('UseParallel','always');
            options.UseParallel = true;
            if z == 1
                lambda = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1];
                [~,fitInfo] = lasso(features(1:end,:)',targets','CV',5,'Lambda',lambda,'Options',options,'Alpha',alpha,'Standardize',false);
                lambdaWhole = fitInfo.LambdaMinMSE;
            end
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
            for i = 1 : CVO.NumTestSets
                for j = 1 : 100
                    c(i,j) = corr(targetPattern(:,j),B{i});
                end
            end
            patternCorrWhole(cLoop,pLoop,z) = mean2(c);
            v = var(cell2mat(B),0,2);
            v = v(v~=0);
            patternVarWhole(cLoop,pLoop,z) = mean(v);
            tsACCWhole(cLoop,pLoop,z) = mean(tsAcc);
            trACCWhole(cLoop,pLoop,z) = mean(trAcc);
            tmp = tril(corr(cell2mat(B)),-1);
            tmp = tmp(tmp~=0);
            coefCorrWhole(cLoop,pLoop,z) = mean(tmp);
            % Cov multipication
            for runNum = 1 : foldNum
                Bbar{runNum} = covMat{runNum}*B{runNum};
            end
            for i = 1 : CVO.NumTestSets
                for j = 1 : 100
                    c(i,j) = corr(targetPattern(:,j),Bbar{i});
                end
            end
            patternCorrCov(cLoop,pLoop,z) = mean2(c);
            v = var(cell2mat(Bbar),0,2);
            v = v(v~=0);
            patternVarCov(cLoop,pLoop,z) = mean(v);
            tmp = tril(corr(cell2mat(Bbar)),-1);
            tmp = tmp(tmp~=0);
            coefCorrCov(cLoop,pLoop,z) = mean(tmp);
            % DCT
            wholeTrials2DGRAD = zeros(trialNum,11,10,timeNum);
            for i = 1 : trialNum
                for j = 1 : 204
                    if labelMap{j,1}(7) == '2'
                        wholeTrials2DGRAD(i,labelMap{j,2}(1),labelMap{j,2}(2),:) = trials(i,j,:);
                    end
                end
            end
            coefNum = [5,5,25];
            n = coefNum(1)*coefNum(2)*coefNum(3);
            featuresGrad = zeros(n,trialNum);
            DGRAD = zeros(size(wholeTrials2DGRAD));
            for i = 1 : trialNum
                DGRAD(i,:,:,:) = shiftdim(dct3(squeeze(wholeTrials2DGRAD(i,:,:,:))),-1);
                featuresGrad(:,i) = reshape(squeeze(DGRAD(i,1:coefNum(1),1:coefNum(2),1:coefNum(3))),n,1);
            end
            [featuresGrad] = mapstd(featuresGrad);
            % Parameter Estimation
            options = statset('UseParallel','always');
            options.UseParallel = true;
            if z == 1
                lambda = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1];
                [~,fitInfo] = lasso(featuresGrad(1:end,:)',targets','CV',5,'Lambda',lambda,'Options',options,'Alpha',alpha,'Standardize',false);
                lambdaDCT = fitInfo.LambdaMinMSE;
            end
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
            tsACCDCT(cLoop,pLoop,z) = mean(tsAcc);
            trACCDCT(cLoop,pLoop,z) = mean(trAcc);
            tmp = tril(corr(cell2mat(BDCT)),-1);
            tmp = tmp(tmp~=0);
            coefCorrDCT(cLoop,pLoop,z) = mean(tmp);
            tmp = tril(corr(cell2mat(tmp1)),-1);
            tmp = tmp(tmp~=0);
            coefCorrDCT1(cLoop,pLoop,z) = mean(tmp);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                    Xi'an jiaotong university
%--------------------------------------------------------------------------
% Author: Zhipeng Ma, December 2021
% Reference: 
%     Zhipeng Ma, Ming Zhao, Chao Gou, "Health Monitoring of Rotating Machinery 
%     Using Probabilistic Time Series Modeln", IEEE TIM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clear
clc


rng('default');
rng(1);

pos3 = [500        400        300          200];
pos2 = [500        400        700          200];
pos1 = [500        400        500          200];


%---------------------------------Parameter--------------------------------
fs = 10000;
k = 1;          
N_train = 5000;    
N_test = 10000;    
%--------------------------------------------------------------------------
[X,Y] = kafbox_data(struct('name','encoder','horizon',k,'embedding',17,'N',20000));
X_train = X(1:N_train,:);
Y_train = Y(1:N_train);
X_test = X(1:N_test,:);
Y_test = Y(1:N_test);

%% 
[sigma_est,c_est,lambda_est] = PTSM_parameter_estimation(X_train,Y_train,1,0.01);
kaf = PTSM(struct('lambda',lambda_est,'M',100,'sn2',c_est,'kerneltype','Multikernel','kernelpar',sigma_est));
Y_est = zeros(N_test,1);
tic
for iii = 1:N_train
        x = X_train(iii,:); y = Y_train(iii); 
        kaf.train(x,y);
        [Y_est, V_est] = kaf.evaluate(X_test);
end
toc
col = {'k',[.5,.5,0.5],[0,.5,0],'m',[0,.75,.75],[.7,0,.5]};
time = (1:length(Y_est))'/fs;
Y_testr = Y_test+600;
Y_estr = Y_est+600;
figure; box on; hold all; 
plot(time,Y_testr); plot(time,Y_estr);
fill([time;flipud(time)],[Y_estr+V_est;flipud(Y_estr-V_est)],...
       col{2},'EdgeColor',col{2},'FaceAlpha',0.1,'EdgeAlpha',0.3);
xlabel('Time [s]'); ylabel('IAS [rpm]');
legend('Raw signals','Predicted mean','Confidence interval');



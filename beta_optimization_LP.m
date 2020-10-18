function [ beta_0 ] = beta_optimization_LP( n_beta,beta_0,v,l_matrix,data_label,n_sample,eig_tol,tol_set,mo )
L_beta=cell(1,n_beta);
for i=1:n_beta
    L_beta{i}=v*diag(l_matrix(:,i))*v';
end
lambda=sum(repmat(beta_0,[n_sample 1]).*l_matrix,2)+eig_tol; % N x 1 (sum P)
if mo==1
    cL_inv=v*diag(1./lambda)*v'; % inverse of cL
    gradient_0=zeros(1,n_beta);
    for i=1:n_beta
        gradient_0(i)=0.5*trace(L_beta{i}*cL_inv)-0.5*data_label'*L_beta{i}*data_label;
    end
else
    gradient_0=zeros(1,n_beta);
    for i=1:n_beta
        gradient_0(i)=0.5*sum(l_matrix(:,i)./lambda)-0.5*data_label'*L_beta{i}*data_label;
    end
end
%%=======================================
%% FW starts
LP_A=-l_matrix;
LP_b=zeros(n_sample,1)+eig_tol;
LP_lb=zeros(n_beta,1)-1e2;
LP_ub=zeros(n_beta,1)+1e2;
options = optimoptions('linprog','Display','none','Algorithm','interior-point'); % linear program (LP) setting for Frank-Wolfe algorithm
% options = optimoptions('linprog','Algorithm','interior-point'); % linear program (LP) setting for Frank-Wolfe algorithm
f=-gradient_0;
FW_beta_direction = linprog(f,LP_A,LP_b,[],[],LP_lb,LP_ub,options); %LP
% while isempty(FW_beta_direction)
%     LP_b=LP_b*1e1;
%     FW_beta_direction = linprog(f,LP_A,LP_b,[],[],LP_lb,LP_ub,options); %LP
% end
while isempty(FW_beta_direction) == 1
    disp('===trying with larger OptimalityTolerance===');
    options.OptimalityTolerance = options.OptimalityTolerance*1e1;
    FW_beta_direction = linprog(f,LP_A,LP_b,[],[],LP_lb,LP_ub,options); %LP
end
FW_beta_direction=FW_beta_direction';
% lll=zeros(length(0:0.01:1),1);
% counter=0;
% for value_try=0:0.01:1
%     counter=counter+1;
% lll(counter)=log_marginal_likelihood(data_label,beta_0+value_try*(FW_beta_direction-beta_0),v,l_matrix,eig_tol,n_sample); % log marginal likelihood
% end
% plot(1:length(0:0.01:1),lll);
obj_net=Inf;
obj_first_round=-log_marginal_likelihood(data_label,beta_0+0*(FW_beta_direction-beta_0),v,l_matrix,eig_tol,n_sample);
iter=0;
while obj_net>1e-2
    iter=iter+1;
    if iter>200
        break
    end
    
    [ alpha_0 ] = GSL_LP_stepsize( ...
        beta_0,...
        FW_beta_direction,...
        l_matrix,...
        eig_tol,...
        L_beta,...
        data_label,...
        n_beta,...
        2);
    
    beta_0_temp=beta_0+alpha_0*(FW_beta_direction-beta_0);
    lambda_check=sum(repmat(beta_0_temp,[n_sample 1]).*l_matrix,2)+eig_tol;
    
    while length(find(lambda_check>0))<n_sample
        alpha_0=alpha_0*(1-1e-5);
        beta_0_temp=beta_0+alpha_0*(FW_beta_direction-beta_0);
        lambda_check=sum(repmat(beta_0_temp,[n_sample 1]).*l_matrix,2)+eig_tol;
    end
    
    beta_0=beta_0+alpha_0*(FW_beta_direction-beta_0);
%         disp(['iter: ' num2str(iter) ' | beta: ' num2str(beta_0) ' | obj: ' num2str(obj_first_round) ' | gradient: ' num2str(gradient_0) ' | step_size: ' num2str(alpha_0)]);
    obj_second_round=-log_marginal_likelihood(data_label,beta_0,v,l_matrix,eig_tol,n_sample);
    obj_net=norm(obj_second_round-obj_first_round);
    obj_first_round=obj_second_round;
    lambda=sum(repmat(beta_0,[n_sample 1]).*l_matrix,2)+eig_tol; % N x 1 (sum P)
    gradient_0=zeros(1,n_beta);
    for i=1:n_beta
        gradient_0(i)=0.5*sum(l_matrix(:,i)./lambda)-0.5*data_label'*L_beta{i}*data_label;
    end
    f=-gradient_0;
    FW_beta_direction = linprog(f,LP_A,LP_b,[],[],LP_lb,LP_ub,options); %LP
    %     while isempty(FW_beta_direction)
    %         LP_b=LP_b*1e1;
    %         FW_beta_direction = linprog(f,LP_A,LP_b,[],[],LP_lb,LP_ub,options); %LP
    %     end
    while isempty(FW_beta_direction) == 1
        disp('===trying with larger OptimalityTolerance===');
        options.OptimalityTolerance = options.OptimalityTolerance*1e1;
        FW_beta_direction = linprog(f,LP_A,LP_b,[],[],LP_lb,LP_ub,options); %LP
    end

    FW_beta_direction=FW_beta_direction';
end
%% FW ends
%%=======================================
lml0=log_marginal_likelihood(data_label,beta_0,v,l_matrix,eig_tol,n_sample);
disp(['converged at iter: ' num2str(iter) ' | beta: ' num2str(beta_0) ' | obj: ' num2str(lml0)]);
end


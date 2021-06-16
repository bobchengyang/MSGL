function [ beta_0 ] = beta_optimization_faster( ldp,n_beta,beta_0,v,l_matrix,data_label,n_sample,eig_tol,tol_set,mo,L,options )

a=1e2;
% iter=Inf;
% while iter>1e2
    
    L_beta=cell(1,n_beta);
    for i=1:n_beta
        L_beta{i}=v*diag(l_matrix(:,i))*v';
    end
    lambda=sum(repmat(beta_0,[n_sample 1]).*l_matrix,2)+eig_tol; % N x 1 (sum P)
    if mo==1
        cL_inv=v*diag(1./lambda)*v'; % inverse of cL
        gradient_0=zeros(1,n_beta);
        for i=1:n_beta
            gradient_0(i)=ldp*trace(L_beta{i}*cL_inv);%-trace(data_label'*L_beta{i}*data_label);
        end
    else
        gradient_0=zeros(1,n_beta);
        for i=1:n_beta
            gradient_0(i)=ldp*sum(l_matrix(:,i)./lambda);%-trace(data_label'*L_beta{i}*data_label);
        end
    end
    %%=======================================
    %% FW starts
    LP_A=-l_matrix;
    LP_b=zeros(n_sample,1)+eig_tol;
    LP_lb=zeros(n_beta,1)-a;
    LP_ub=zeros(n_beta,1)+a;
    LP_Aeq=zeros(1,n_beta);
    for i=1:n_beta
        LP_Aeq(i)=trace(data_label'*(L^i)*data_label);
    end    
    LP_beq=n_sample;

    f=-gradient_0;
    FW_beta_direction = linprog(f,LP_A,LP_b,LP_Aeq,LP_beq,LP_lb,LP_ub,options); %LP

    while isempty(FW_beta_direction) == 1
        disp('===trying with larger OptimalityTolerance===');
        options.ConstraintTolerance = options.ConstraintTolerance*1e1;
        options.OptimalityTolerance = options.OptimalityTolerance*1e1;
        FW_beta_direction = linprog(f,LP_A,LP_b,LP_Aeq,LP_beq,LP_lb,LP_ub,options); %LP
    end
    FW_beta_direction=FW_beta_direction';

    obj_net=Inf;
%     [obj_first_round,obj_term1]=log_marginal_likelihood_instant(ldp,data_label,beta_0+0*(FW_beta_direction-beta_0),v,l_matrix,eig_tol,n_sample);
    obj_first_round=-ldp*sum(log(lambda));
    iter=0;
    GD_NR=2;
    while obj_net>1e-2 && iter<=1e2
        iter=iter+1;
%         if iter>1e2
%             if GD_NR==2
%                 iter=0;
%                 options.OptimalityTolerance = 1e-5;
%                 obj_net=Inf;
%                 alpha_0=0;
%                 [obj_first_round,obj_term1]=log_marginal_likelihood_instant(ldp,data_label,beta_0+0*(FW_beta_direction-beta_0),v,l_matrix,eig_tol,n_sample);
% %                 obj_first_round=-ldp*sum(log(lambda));
%                 GD_NR=1;%try gradient descent to get alpha_0 instead of NR
%                 %break
%             else
%                 beta_0=zeros(1,n_beta);
%                 a=a/1e1;
%                 break
%             end
%         end
        
        [ alpha_0 ] = GSL_LP_stepsize_instant(ldp, ...
            beta_0,...
            FW_beta_direction,...
            l_matrix,...
            eig_tol,...
            L_beta,...
            data_label,...
            n_beta,...
            GD_NR);
        
        %alpha_0
        beta_0_temp=beta_0+alpha_0*(FW_beta_direction-beta_0);
        lambda_check=sum(repmat(beta_0_temp,[n_sample 1]).*l_matrix,2)+eig_tol;
        
        while length(find(lambda_check>0))<n_sample
            alpha_0=alpha_0*(1-1e-5);
            beta_0_temp=beta_0+alpha_0*(FW_beta_direction-beta_0);
            lambda_check=sum(repmat(beta_0_temp,[n_sample 1]).*l_matrix,2)+eig_tol;
        end
        
%         disp(['iter: ' num2str(iter) ' | beta: ' num2str(beta_0) ' | obj: ' num2str(obj_first_round) '/' num2str(obj_term1) ' | gradient: ' num2str(gradient_0) ' | step_size: ' num2str(alpha_0)]);       
        
        beta_0=beta_0+alpha_0*(FW_beta_direction-beta_0);
  
        lambda=sum(repmat(beta_0,[n_sample 1]).*l_matrix,2)+eig_tol; % N x 1 (sum P)
        obj_second_round=-ldp*sum(log(lambda));
%         [obj_second_round,obj_term1]=log_marginal_likelihood_instant(ldp,data_label,beta_0,v,l_matrix,eig_tol,n_sample);

        obj_net=norm(obj_second_round-obj_first_round);
        
        obj_first_round=obj_second_round;
        
        
%         lambda=lambda/max(lambda); % not used in ICASSP submission
        gradient_0=zeros(1,n_beta);
        for i=1:n_beta
            gradient_0(i)=sum(l_matrix(:,i)./lambda);%-trace(data_label'*L_beta{i}*data_label);
        end
        f=-gradient_0;
        FW_beta_direction = linprog(f,LP_A,LP_b,LP_Aeq,LP_beq,LP_lb,LP_ub,options); %LP

        while isempty(FW_beta_direction) == 1
            disp('===trying with larger OptimalityTolerance===');
        options.ConstraintTolerance = options.ConstraintTolerance*1e1;
        options.OptimalityTolerance = options.OptimalityTolerance*1e1;
            FW_beta_direction = linprog(f,LP_A,LP_b,LP_Aeq,LP_beq,LP_lb,LP_ub,options); %LP
        end
        
        FW_beta_direction=FW_beta_direction';
    end
    
% end
%% FW ends
%%=======================================
% beta_0=FW_beta_direction;
% [lml0,obj_term1]=log_marginal_likelihood_instant(ldp,data_label,beta_0,v,l_matrix,eig_tol,n_sample);
disp(['converged at iter: ' num2str(iter) ' | beta: ' num2str(beta_0) ' | obj: ' num2str(obj_second_round) '/' num2str(n_sample)]);
end


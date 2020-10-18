function [ beta_0 ] = beta_optimization( n_beta,beta_0,v,l_matrix,data_label,n_sample,eig_tol,tol_set,mo )

lml0=log_marginal_likelihood(data_label,beta_0,v,l_matrix,eig_tol,n_sample); % log marginal likelihood

tol=Inf;
iter=0;
step_size=1e1; % initial step size

L_beta=cell(1,n_beta);
for i=1:n_beta
    L_beta{i}=v*diag(l_matrix(:,i))*v';
end

while tol>tol_set
    iter=iter+1;
    if mod(iter,1000)==0
        disp(['iter: ' num2str(iter) ' | beta: ' num2str(beta_0_temp) ' | obj: ' num2str(lml0)]);
    end
    lambda=sum(repmat(beta_0,[n_sample 1]).*l_matrix,2)+eig_tol;
    if mo==1
        cL_inv=v*diag(1./lambda)*v'; % inverse of cL
        gradient_0=zeros(1,n_beta);
%         hessian_0=zeros(n_beta);
        for i=1:n_beta
%             L_beta=v*diag(l_matrix(:,i))*v';
            gradient_0(i)=0.5*trace(L_beta{i}*cL_inv)-0.5*data_label'*L_beta{i}*data_label;
        end
%         gradient_0=[0.5*trace(L*cL_inv)-0.5*data_label'*L*data_label...
%             0.5*trace(L2*cL_inv)-0.5*data_label'*L2*data_label...
%             0.5*trace(L3*cL_inv)-0.5*data_label'*L3*data_label...
%             0.5*trace(L4*cL_inv)-0.5*data_label'*L4*data_label];
    else
        gradient_0=zeros(1,n_beta);
%         hessian_0=zeros(n_beta);
        for i=1:n_beta
%             L_beta=v*diag(l_matrix(:,i))*v';
            gradient_0(i)=0.5*sum(l_matrix(:,i)./lambda)-0.5*data_label'*L_beta{i}*data_label;
% gradient_0(i)=-0.5*data_label'*L_beta{i}*data_label;
%             for j=1:n_beta
%             hessian_0(i,j)=-0.5*sum(l_matrix(:,i).*l_matrix(:,j)./lambda.^2);
%             end
        end
%         disp(['min hessian eig: ' num2str(max(eig(hessian_0)))]);
%         gradient_0=[0.5*sum(l1./lambda)-0.5*data_label'*L*data_label...
%             0.5*sum(l2./lambda)-0.5*data_label'*L2*data_label...
%             0.5*sum(l3./lambda)-0.5*data_label'*L3*data_label...
%             0.5*sum(l4./lambda)-0.5*data_label'*L4*data_label];
    end
    disp(['iter: ' num2str(iter) ' | beta: ' num2str(beta_0) ' | obj: ' num2str(lml0) ' | gradient: ' num2str(gradient_0) ' | step_size: ' num2str(step_size)]);

    beta_0_temp=beta_0+step_size*gradient_0; % gradient ascent

%     b1b2=beta_0_temp(1)/beta_0_temp(2);  % beta_1/beta_2
    lambda_check=sum(repmat(beta_0_temp,[n_sample 1]).*l_matrix,2)+eig_tol;
%     disp(['beta_temp: ' num2str(beta_0_temp)]);
    
    while length(find(lambda_check>0))<n_sample
    %while (beta_0_temp(2)<0 && b1b2>=-l1m) || (beta_0_temp(2)>0 && b1b2<=-l12) % not PD
        step_size=step_size/2; % decrease the step size to find a larger objective
        beta_0_temp=beta_0+step_size*gradient_0;
%         disp(['trying beta_temp: ' num2str(beta_0_temp)]);
%         b1b2=beta_0_temp(1)/beta_0_temp(2);
        lambda_check=sum(repmat(beta_0_temp,[n_sample 1]).*l_matrix,2)+eig_tol;
    end
    
    beta_0=beta_0_temp; % update beta's
    lml0_c=log_marginal_likelihood(data_label,beta_0,v,l_matrix,eig_tol,n_sample); % log marginal likelihood
    tol=norm(lml0_c-lml0);
    lml0=lml0_c;
    step_size=step_size*1.01; % increase the step size to converge faster
end
beta_0=beta_0/norm(beta_0)*n_beta;
lml0=log_marginal_likelihood(data_label,beta_0,v,l_matrix,eig_tol,n_sample);
disp(['converged at iter: ' num2str(iter) ' | beta: ' num2str(beta_0_temp) ' | obj: ' num2str(lml0)]);
end


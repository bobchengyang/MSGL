function [ beta_0 ] = beta_optimization_oaat( L,L2,beta_0,v,l1,l2,l1m,l12,data_label,n_sample,eig_tol,tol_set,mo )

lml0=log_marginal_likelihood(data_label,beta_0,v,l1,l2,eig_tol,n_sample); % log marginal likelihood
lml0_outer=lml0;


tol_outer=Inf;
while tol_outer>tol_set
    for beta_i=1:2
        disp('================================');
        disp(['updating beta ' num2str(beta_i)]);
        tol_inner=Inf;
        step_size=.1/n_sample; % initial step size
        iter=0;
        while tol_inner>tol_set
            iter=iter+1;
                disp(['iter: ' num2str(iter) ' | beta: ' num2str(beta_0) ' | obj: ' num2str(lml0)]);
%             if mod(iter,1000)==0
%                 disp(['iter: ' num2str(iter) ' | beta: ' num2str(beta_0_temp) ' | obj: ' num2str(lml0)]);
%             end
            lambda=beta_0(1)*l1+beta_0(2)*l2+eig_tol; % eigenvalues of the cL
            if mo==1
                cL_inv=v*diag(1./lambda)*v'; % inverse of cL
                gradient_0=[0.5*trace(L*cL_inv)-0.5*data_label'*L*data_label...
                    0.5*trace(L2*cL_inv)-0.5*data_label'*L2*data_label];
            else
                gradient_0=[0.5*sum(l1./lambda)-0.5*data_label'*L*data_label...
                    0.5*sum(l2./lambda)-0.5*data_label'*L2*data_label];
            end
            
            beta_0_temp=beta_0;
            beta_0_temp(beta_i)=beta_0(beta_i)+step_size*gradient_0(beta_i); % gradient ascent
            b1b2=beta_0_temp(1)/beta_0_temp(2);  % beta_1/beta_2
            
            while (beta_0_temp(2)<0 && b1b2>=-l1m) || (beta_0_temp(2)>0 && b1b2<=-l12) % not PD
                step_size=step_size/2; % decrease the step size to find a larger objective
                beta_0_temp(beta_i)=beta_0(beta_i)+step_size*gradient_0(beta_i);
                b1b2=beta_0_temp(1)/beta_0_temp(2);
            end
            
            beta_0=beta_0_temp; % update beta's
            lml0_c=log_marginal_likelihood(data_label,beta_0,v,l1,l2,eig_tol,n_sample); % log marginal likelihood
            tol_inner=norm(lml0_c-lml0);
            lml0=lml0_c;
            step_size=step_size*1.01; % increase the step size to converge faster
        end
        disp(['converged at iter: ' num2str(iter) ' | beta: ' num2str(beta_0_temp) ' | obj: ' num2str(lml0)]);
    end
    tol_outer=norm(lml0_c-lml0_outer);
    lml0_outer=lml0_c;
end

end


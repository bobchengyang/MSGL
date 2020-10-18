clear;clc;close all;

load L_data_label_rand_0_diabetes.mat

n_sample=size(L,1); % number of tested samples
eig_tol=1e-8;
tol_set=1e-5;
mo=1; % trace (1) or eigenvalue (2) mode
[ beta_0 ] = beta_optimization( L,data_label,n_sample,eig_tol,tol_set,mo );
% disp(['beta: ' num2str(beta_0)]);
test_length=461; % number of samples to be tested

L=L(1:test_length,1:test_length); % graph Laplacian of the feature

data_label=data_label(1:test_length);

n_sample=size(L,1); % number of tested samples

zz=logical(tril(ones(n_sample),-1)); % indices of the lower triangular entries of L
dia_idx=1:n_sample+1:n_sample^2; % indices of the diagonal entries

L_offdia=L;
L_offdia(dia_idx)=0;
L_offdia=-sum(L_offdia,2);
D=diag(L_offdia);

L(dia_idx)=L_offdia;
% L=D^(1/2)*L*D^(1/2);
% L=L+(abs(min(eig(L)))*eye(n_sample));

[v,d]=eig(L);
l1=diag(d);
l1(1)=0;
l12=l1(2);
l1m=l1(end);

L2=L*L; % Laplacian to the power of 2
L2=(L2+L2')/2;
[v2,d2]=eig(L2);
l2=diag(d2);
l2(1)=0;
l22=l2(2);
% %=======Graph classifier starts======
% cvx_begin sdp
% variable x(3,1);
% minimize(x(1))
% subject to
% x(1)*L^0+x(2)*L+x(3)*L2>=0;
% cvx_end
%========Graph classifier ends=======

% options = optimoptions('linprog','Algorithm','interior-point'); % linear program (LP) setting for Frank-Wolfe algorithm
options = optimoptions('linprog','Display','none','Algorithm','interior-point'); % linear program (LP) setting for Frank-Wolfe algorithm
f=1; % linear programming objective, i.e., find the minimum beta

% beta_0=[1 10]; % initial beta's
% B=beta_0(1)*L+beta_0(2)*L2; % initial matrix B, i.e., the initial inverse covariance matrix
% B=(B+B')/2; % make sure B is symmetric
% B=B+(abs(min(eig(B))+1e-8)*eye(n_sample));
% cond(B)
%% gradient
% gradient_0=[0.5*trace((B*data_label*(B*data_label)'-B)*L);...
%             0.5*trace((B*data_label*(B*data_label)'-B)*L2)];

beta_0=[1 1]; % initial beta's
% B=beta_0(1)*L+beta_0(2)*L2;
% B=(B+B')/2;
% [vb,db]=eig(B);
% B=B+(abs(min(eig(B))+1e-8)*eye(n_sample));
lml0=log_marginal_likelihood(data_label,beta_0,v,l1,l2,eig_tol); % log marginal likelihood
tol=Inf;
step_size=.1/test_length;
flag=1;

B=-19*L+148*L2;
si=100;
lml_map=zeros(si,si);
for i=(1:si)-50
    for j=(1:si)-50
        if (i/j>-l12 && j>0) || (i/j<-l1m && j<0) || (i>0 && j==0)
        beta_0(1)=i;
        beta_0(2)=j;
%         B=beta_0(1)*L+beta_0(2)*L2;
%         if min(eig(B))<=0
%         B=B+(abs(min(eig(B))+1e-8)*eye(n_sample));
%         else
%             B=B+1e-8*eye(n_sample);
%         end
%         lml0=-0.5*data_label'*B*data_label-0.5*log(det(inv(B)));
        lml0=log_marginal_likelihood(data_label,beta_0,v,l1,l2,eig_tol); % log marginal likelihood
        lml_map(i+50,j+50)=lml0;
        else
        lml_map(i+50,j+50)=NaN;    
        end
    end
end

lml_map_idx=lml_map==real(lml_map);
lml_map(~lml_map_idx)=NaN;
% lml_map(lml_map==Inf)=NaN;
imagesc(lml_map);axis equal;
[a,b]=max(lml_map(:));
disp(['max row: ' num2str(mod(b,si)) ' | max column: ' num2str(ceil(b/si))]);
disp(['initial lml: ' num2str(lml0) ' | min eig B: ' num2str(min(eig(B)))]);


iter=0;
while tol>1e-5
    iter=iter+1;
    if mod(iter,1000)==0
        disp(['iter: ' num2str(iter) ' | beta: ' num2str(beta_0_temp) ' | obj: ' num2str(lml0)]);
    end
    lambda=beta_0(1)*l1+beta_0(2)*l2+eig_tol;
    % B=v*diag(lambda)*v';
    cL_inv=v*diag(1./lambda)*v';
    gradient_0=[0.5*trace(L*cL_inv)-0.5*data_label'*L*data_label...
                0.5*trace(L2*cL_inv)-0.5*data_label'*L2*data_label];

    beta_0_temp=beta_0+step_size*gradient_0;

    b1b2=beta_0_temp(1)/beta_0_temp(2); 
    
    while (beta_0_temp(2)<0 && b1b2>=-l1m) || (beta_0_temp(2)>0 && b1b2<=-l12) % not PD
        step_size=step_size/2;
        beta_0_temp=beta_0+step_size*gradient_0;
        b1b2=beta_0_temp(1)/beta_0_temp(2); 
    end

        beta_0=beta_0_temp;
        lambda=beta_0(1)*l1+beta_0(2)*l2+eig_tol;
        B=v*diag(lambda)*v';
        B=(B+B')/2;

        lml0_c=log_marginal_likelihood(data_label,beta_0,v,l1,l2,eig_tol); % log marginal likelihood
        tol=norm(lml0_c-lml0);
        lml0=lml0_c;
        step_size=step_size*1.01;
   
end

% iter=0;
% while tol>1e-5
%     iter=iter+1;
%     if mod(iter,1000)==0
%         disp(['iter: ' num2str(iter) ' | beta: ' num2str(beta_0_temp) ' | obj: ' num2str(lml0)]);
%     end
%     lambda=beta_0(1)*l1+beta_0(2)*l2+1e-8;
%     % B=v*diag(lambda)*v';
%     cL_inv=v*diag(1./lambda)*v';
%     gradient_0=[0.5*trace(L*cL_inv)-0.5*data_label'*L*data_label...
%         0.5*trace(L2*cL_inv)-0.5*data_label'*L2*data_label];
%     if flag==1
%     beta_0_temp=beta_0+step_size*gradient_0;
%     else
%     beta_0_temp=beta_0_temp+step_size*gradient_0;     
%     end
%     b1b2=beta_0_temp(1)/beta_0_temp(2);
% %     disp(['obj: ' num2str(lml0) ' | beta: ' num2str(beta_0_temp) ' | lambda: ' num2str([l12 l1m]) ' | b1b2: ' num2str(b1b2) ' | gradient: ' num2str(gradient_0)]);
%     
%     if beta_0_temp(2)<0 && b1b2>=-l1m
% %         disp('cL is not PD!');
%         flag=0;
%         beta_0_temp(2)=-abs(beta_0_temp(1)/l1m)+1e-2;
%         beta_0=beta_0_temp;
%                 lambda=beta_0(1)*l1+beta_0(2)*l2+1e-8;
%         B=v*diag(lambda)*v';
% %         B=beta_0(1)*L+beta_0(2)*L2; % initial matrix B, i.e., the initial inverse covariance matrix
%         B=(B+B')/2;
%         if min(eig(B))<0
%             asdf=1;
%         end
%     elseif beta_0_temp(2)>0 && b1b2<=-l12
% %         disp('cL is not PD!');
%         flag=0;
%         beta_0_temp(2)=abs(beta_0_temp(1)/l1m)+1e-2;
%         beta_0=beta_0_temp;
%         lambda=beta_0(1)*l1+beta_0(2)*l2+1e-8;
%         B=v*diag(lambda)*v';
% %         B=beta_0(1)*L+beta_0(2)*L2; % initial matrix B, i.e., the initial inverse covariance matrix
%         B=(B+B')/2;
%         if min(eig(B))<0
%             asdf=1;
%         end
%     else
% %         disp('cL is PD');
%         beta_0=beta_0_temp;
%         lambda=beta_0(1)*l1+beta_0(2)*l2+1e-8;
%         B=v*diag(lambda)*v';
% %         B=beta_0(1)*L+beta_0(2)*L2; % initial matrix B, i.e., the initial inverse covariance matrix
%         B=(B+B')/2;
%         if min(eig(B))<0
%             asdf=1;
%         end
%         lml0_c=log_marginal_likelihood(data_label,beta_0,v,l1,l2); % log marginal likelihood
%         tol=norm(lml0_c-lml0);
%         lml0=lml0_c;
%         flag=1;
%     end
% end

lml_map_idx=lml_map==real(lml_map);
lml_map(~lml_map_idx)=NaN;
lml_map(lml_map==Inf)=NaN;
imagesc(lml_map);
[a,b]=max(lml_map(:));
disp(['max row: ' num2str(mod(b,si)) ' | max column: ' num2str(ceil(b/si))]);
disp(['initial lml: ' num2str(lml0) ' | min eig B: ' num2str(min(eig(B)))]);

while tol>1e-3
    lambda=beta_0(1)*l1+beta_0(2)*l2;
    % B=v*diag(lambda)*v';
    cL_inv=v*diag(1./lambda)*v';
    gradient_0=[0.5*trace(L*cL_inv)-0.5*data_label'*L*data_label...
        0.5*trace(L2*cL_inv)-0.5*data_label'*L2*data_label];
    if flag==1
    beta_0_temp=beta_0+step_size*gradient_0;
    else
    beta_0_temp=beta_0_temp+step_size*gradient_0;     
    end
    b1b2=beta_0_temp(1)/beta_0_temp(2);
    disp(['obj: ' num2str(lml0) ' | beta: ' num2str(beta_0_temp) ' | lambda_o: ' num2str(l12) ' | b1b2: ' num2str(b1b2) ' | gradient: ' num2str(gradient_0)]);
    
    if beta_0_temp(2)<0 && b1b2>-l12
        disp('cL is not PD!');
        flag=0;
        beta_0=beta_0_temp;
        B=beta_0(1)*L+beta_0(2)*L2; % initial matrix B, i.e., the initial inverse covariance matrix
        B=(B+B')/2;
        if min(eig(B))<0
            asdf=1;
        end
    elseif beta_0_temp(2)>0 && b1b2<-l12
        disp('cL is not PD!');
        flag=0;
        beta_0=beta_0_temp;
        B=beta_0(1)*L+beta_0(2)*L2; % initial matrix B, i.e., the initial inverse covariance matrix
        B=(B+B')/2;
        if min(eig(B))<0
            asdf=1;
        end
    else
        disp('cL is PD');
        beta_0=beta_0_temp;
        B=beta_0(1)*L+beta_0(2)*L2; % initial matrix B, i.e., the initial inverse covariance matrix
        B=(B+B')/2;
        if min(eig(B))<0
            asdf=1;
        end
        lml0_c=log_marginal_likelihood(data_label,beta_0,v,l1,l2); % log marginal likelihood
        tol=norm(lml0_c-lml0);
        lml0=lml0_c;
        flag=1;
    end
end

B=-19*L+148*L2;
si=100;
lml_map=zeros(si,si);
for i=(1:si)-50
    for j=(1:si)-50
        B=i*L+j*L2; % initial matrix B, i.e., the initial inverse covariance matrix
        B=(B+B')/2; % make sure B is symmetric
        lml0=log_marginal_likelihood(B+(abs(min(eig(B))+1e-8)*eye(n_sample)),data_label); % log marginal likelihood
        lml_map(i+50,j+50)=lml0;
    end
end
lml_map_idx=lml_map==real(lml_map);
lml_map(~lml_map_idx)=NaN;
lml_map(lml_map==Inf)=NaN;
imagesc(lml_map);
[a,b]=max(lml_map(:));
disp(['max row: ' num2str(mod(b,si)) ' | max column: ' num2str(ceil(b/si))]);
disp(['initial lml: ' num2str(lml0) ' | min eig B: ' num2str(min(eig(B)))]);

tol=1e-2;
rho=1e-5;

counter=0;
for beta_i=2:-1:0
    current_tol=Inf;
    while current_tol>tol
        
        L_beta_i=L^(beta_i);% current Laplacian: Laplacian to the power of beta_i
        
        [~,remove_mask]=BFS_Balanced(-B); % check graph balance, get the locations of the removed entries
        remove_mask=logical(remove_mask); % logicalize the removed entries map
        remove_mask=~remove_mask; % invert it
        
        B=-BFS_Balanced(-B); % check graph balance
        
        % now matrix B is indeed a balanced signed graph (if it is a signed graph)
        
        %         [v,d]=eig(B); a=v(:,1); % eig
        
        if counter==0
            counter=1;
            rng(0);
            [a,~] = lobpcg_fv(randn(n_sample,1),B,1e-4,200); % LOBPCG
        else
            [a,~] = lobpcg_fv(randn(n_sample,1),B,1e-4,200); % LOBPCG
        end
        
        scaled_B = (1./a) .* B .* a'; % scaled B
        sf = (1./a) .* ones(n_sample) .* a'; % scaled factors
        
        %% check the left-ends
        scaled_B_offdia=scaled_B;
        scaled_B_offdia(dia_idx)=0;
        leftEnds=diag(scaled_B)-sum(abs(scaled_B_offdia),2);
        disp(['left ends: ' num2str(leftEnds')]);
        
        B_ncb=B-beta_0(beta_i+1)*L_beta_i; % subtract the components that are with the current beta
        
        B_ncb=B_ncb.*remove_mask; % the removed locations should not be subtracted
        
        if beta_i==0
            t_list=unique(B_ncb(zz)); % get the list of the beta thresholds
        else
            threshold_map=-B_ncb./L_beta_i;
            threshold_map(dia_idx)=0;
            t_list=unique(threshold_map(zz)); % get the list of the beta thresholds
        end
        
        B_ncb_offdia=B_ncb;
        B_ncb_offdia(dia_idx)=0;
        
        sf=abs(sf);
        sf(dia_idx)=0;
        
        beta_candidate=[];
        
        for i=length(t_list):-1:0
            sign_mask=zeros(n_sample);
            
            if beta_i~=0
                if i==length(t_list) % all positive
                    lb=t_list(i);
                    ub=Inf;
                    test_offdia=L_beta_i.*(lb+rho)+B_ncb;
                elseif i==0 % all negative
                    lb=-Inf;
                    ub=t_list(1);
                    test_offdia=L_beta_i.*(ub+rho)+B_ncb;
                else % positive/negative
                    lb=t_list(i);
                    ub=t_list(i+1);
                    test_offdia=L_beta_i.*((lb+ub)/2)+B_ncb;
                end
                sign_mask(test_offdia>0)=1;
                sign_mask(test_offdia<0)=-1;
            else % beta_i==0
                lb=[];
                ub=[];
                sign_mask(B_ncb<0)=-1;
                sign_mask(B_ncb>0)=1;
            end
            
            offdia_b=B_ncb_offdia.*sf.*sign_mask.*remove_mask; % b
            
            offdia_A=sf.*sign_mask.*L_beta_i.*remove_mask; % A
            
            if beta_i==0
                A=-ones(n_sample,1);
            else
                A=sum(offdia_A,2)-diag(L_beta_i);
            end
            
            b=diag(B_ncb)-sum(offdia_b,2)-rho;
            
            try
                x = linprog(f,A,b,[],[],lb,ub,options); %LP
                if mod(i,1000)==0
                    if length(x)>0
                        disp(['Y || solved ' num2str(i/length(t_list))]);
                    end
                end
                beta_candidate=[beta_candidate;x]; %store beta candidate
            catch
                disp('N || no solution');
            end
            
        end
        
        beta_best=min(beta_candidate); % choose the smallest beta candidate as the best beta
        beta_0(beta_i+1)=beta_best; % update beta vector
        B=beta_0(1)*(L^0)+beta_0(2)*L+beta_0(3)*L2; % initial matrix B, i.e., the initial inverse covariance matrix
        B=(B+B')/2; % make sure B is symmetric
        B_balanced=B.*remove_mask;
        
        %% check the validity of the results
        leftEnds=diag(B)-sum(abs(B_balanced.*sf),2);
        disp(['left ends: ' num2str(leftEnds')]);
        
        lml=log_marginal_likelihood(B_balanced,data_label,test_length); % get the current log marginal likelihood
        current_tol=norm(lml-lml0); % get the current tol
        disp(['lml: ' num2str(lml) ' | current tol: ' num2str(current_tol)...
            ' | min eig B: ' num2str(min(eig(B_balanced))) ' | beta: ' num2str(beta_best)]);
        lml0=lml;
    end
end

disp(['min eig B: ' num2str(min(eig(B)))]);
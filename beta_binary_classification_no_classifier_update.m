function [beta_0_test,initial_obj_term1,obj_term2] = ...
    beta_binary_classification_no_classifier_update(ldp,beta_0_current,dataset_i,n_beta, ...
    feature_train_test, ...
    x_valid, ...
    M,options)

[n_sample_full, n_feature]= size(feature_train_test); %get the number of samples and the number of features
[ x_valid,~ ] = class_num_to_vec( x_valid,n_sample_full );
% n_sample_test=length(class_test);

% M = eye(n_feature);
tol_set=1e-2;
mo=2;
flag=0;
[ L ] = graph_Laplacian_train_test( dataset_i,feature_train_test, M ); % full observation
cL=0;
for i=1:n_beta
    cL=cL+1*L^i;
end
initial_obj_term1=trace(x_valid'*cL*x_valid);
%% get eigen pairs starts
[v,d]=eig(L); % eigen-decomposition of L
l1=diag(d); % get the eigen-values (make sure that L is symmetric)
l1(1)=0; % set the first eigenvalue to 0

%% get eigen pairs ends
% tol_compare=-Inf;
beta_0_current=zeros(1,n_beta); % initial beta's

l_matrix=zeros(n_sample_full,n_beta);
for i=1:n_beta
    l_matrix(:,i)=l1.^i;
end

eig_tol=1e-8;

%     lambda=lambda/max(lambda); % used in ICASSP submission
%     lambda=lambda/norm(lambda);

[ beta_0_test ] = beta_optimization_faster(ldp,n_beta,beta_0_current,v,l_matrix,x_valid,n_sample_full,eig_tol,tol_set,mo,L,options );
lambda=sum(repmat(beta_0_test,[n_sample_full 1]).*l_matrix,2)+eig_tol;
lambda(lambda<0)=eig_tol;
cL=0;
for i=1:n_beta
    cL=cL+beta_0_test(i)*L^i;
end
obj_term1=n_sample_full;
obj_term2=-ldp*sum(log(lambda));
disp(['obj right after beta learning: ' num2str(obj_term1) '/' num2str(obj_term2)]);
% % % obj_term1=trace(x_valid'*cL*x_valid);
% % % obj_term2=-ldp*sum(log(lambda));
% % % disp(['obj right after beta learning: ' num2str(obj_term1) '/' num2str(obj_term1+obj_term2)]);
end
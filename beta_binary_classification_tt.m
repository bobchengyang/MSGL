function [error_classifier] = ...
    beta_binary_classification_tt( class_test, ...
    feature_train_test, ...
    initial_label_index, ...
    class_train_test, ...
    classifier_i)

[n_sample_full, n_feature]= size(feature_train_test); %get the number of samples and the number of features

M = eye(n_feature);
tol_set=1e-2;
mo=1;

[ L ] = graph_Laplacian_train_test( feature_train_test, M ); % full observation

%% get eigen pairs starts
[v,d]=eig(L); % eigen-decomposition of L
l1=diag(d); % get the eigen-values (make sure that L is symmetric)
l1(1)=0; % set the first eigenvalue to 0
% l12=l1(2); % get the second eigenvalue
% l1m=l1(end); % get the last eigenvalue

% L2=v*d.^2*v'; % Laplacian to the power of 2
% L2=L*L; % Laplacian to the power of 2
% L2=(L2+L2')/2; % make sure that L2 is symmetric
% [v2,d2]=eig(L2);
% l2=diag(d2);
% l2(1)=0;
l2=l1.^2; %
% l22=l2(2);
%% get eigen pairs ends
beta_0=[1 1]; % initial beta's

eig_tol=1e-8;
lambda=beta_0(1)*l1+beta_0(2)*l2+eig_tol;
cL=v*diag(lambda)*v';

if classifier_i==1 % 3-NN classifier
    knn_size = 3;
    %========KNN classifier starts========
    fl = class_train_test(initial_label_index);
    fl(fl == -1) = 0;
    x = KNN(fl, feature_train_test(initial_label_index,:), sqrtm(full(M)), knn_size, feature_train_test(~initial_label_index,:));
    x(x==0) = -1;
    x_valid = class_train_test;
    x_valid(~initial_label_index) = x;
    %=========KNN classifier ends=========
elseif classifier_i==2 % Mahalanobis classifier
    %=======Mahalanobis classifier starts========
    [m,X] = ...
        mahalanobis_classifier_variables(...
        feature_train_test,...
        class_train_test,...
        initial_label_index);
    z=mahalanobis_classifier(m,M,X);
    x_valid=class_train_test;
    x_valid(~initial_label_index)=z;
    %========Mahalanobis classifier ends=========
else % GLR-based classifier
    %=======Graph classifier starts=======
    cvx_begin
    variable x(n_sample_full,1);
    minimize(x'*cL*x)
    subject to
    x(initial_label_index) == class_train_test(initial_label_index);
    cvx_end
    x_valid = sign(x);
    %========Graph classifier ends========
end

diff_label = x_valid - class_train_test;
error_classifier = size(find(diff_label~=0),1)*size(find(diff_label~=0),2)/size(class_test,1);

disp(['objective before beta optimization : ' num2str(x_valid'*cL*x_valid)]);
disp(['error rate before beta optimization : ' num2str(error_classifier)]);

[ L ] = graph_Laplacian_train_test( feature_train_test(initial_label_index,:), M ); % training observation

%% get eigen pairs starts
[v,d]=eig(L); % eigen-decomposition of L
l1=diag(d); % get the eigen-values (make sure that L is symmetric)
l1(1)=0; % set the first eigenvalue to 0
l12=l1(2); % get the second eigenvalue
l1m=l1(end); % get the last eigenvalue

L2=v*d.^2*v'; % Laplacian to the power of 2
% L2=L*L; % Laplacian to the power of 2
% L2=(L2+L2')/2; % make sure that L2 is symmetric
% [v2,d2]=eig(L2);
% l2=diag(d2);
% l2(1)=0;
l2=l1.^2; %
% l22=l2(2);
%% get eigen pairs ends
beta_0=[1 1]; % initial beta's

eig_tol=1e-8;
% lambda=beta_0(1)*l1+beta_0(2)*l2+eig_tol;
% cL=v*diag(lambda)*v';

[ beta_0 ] = beta_optimization( L,L2,beta_0,v,l1,l2,l1m,l12,class_train_test(initial_label_index),n_sample_full,eig_tol,tol_set,mo );

[ L ] = graph_Laplacian_train_test( feature_train_test, M ); % full observation

%% get eigen pairs starts
[v,d]=eig(L); % eigen-decomposition of L
l1=diag(d); % get the eigen-values (make sure that L is symmetric)
l1(1)=0; % set the first eigenvalue to 0
% l12=l1(2); % get the second eigenvalue
% l1m=l1(end); % get the last eigenvalue

% L2=v*d.^2*v'; % Laplacian to the power of 2
% L2=L*L; % Laplacian to the power of 2
% L2=(L2+L2')/2; % make sure that L2 is symmetric
% [v2,d2]=eig(L2);
% l2=diag(d2);
% l2(1)=0;
l2=l1.^2; %
% l22=l2(2);
%% get eigen pairs ends

eig_tol=1e-8;
lambda=beta_0(1)*l1+beta_0(2)*l2+eig_tol;
cL=v*diag(lambda)*v';

if classifier_i==1 % 3-NN classifier
    knn_size = 3;
    %========KNN classifier starts========
    fl = class_train_test(initial_label_index);
    fl(fl == -1) = 0;
    x = KNN(fl, feature_train_test(initial_label_index,:), sqrtm(full(M)), knn_size, feature_train_test(~initial_label_index,:));
    x(x==0) = -1;
    x_valid = class_train_test;
    x_valid(~initial_label_index) = x;
    %=========KNN classifier ends=========
elseif classifier_i==2 % Mahalanobis classifier
    %=======Mahalanobis classifier starts========
    [m,X] = ...
        mahalanobis_classifier_variables(...
        feature_train_test,...
        class_train_test,...
        initial_label_index);
    z=mahalanobis_classifier(m,M,X);
    x_valid=class_train_test;
    x_valid(~initial_label_index)=z;
    %========Mahalanobis classifier ends=========
else % GLR-based classifier
    %=======Graph classifier starts=======
    cvx_begin
    variable x(n_sample_full,1);
    minimize(x'*cL*x)
    subject to
    x(initial_label_index) == class_train_test(initial_label_index);
    cvx_end
    x_valid = sign(x);
    %========Graph classifier ends========
end

disp(['objective after beta optimization: ' num2str(x_valid'*cL*x_valid)]);
disp(['error rate after beta optimization: ' num2str(error_classifier)]);

end
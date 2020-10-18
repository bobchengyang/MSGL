function [error_classifier_current] = ...
    beta_binary_classification( dataset_i,n_beta,class_test, ...
    feature_train_test, ...
    initial_label_index, ...
    class_train_test, ...
    classifier_i)

[n_sample_full, n_feature]= size(feature_train_test); %get the number of samples and the number of features

M = eye(n_feature);
tol_set=1e-2;
mo=1;
flag=0;
[ L ] = graph_Laplacian_train_test( dataset_i,feature_train_test, M ); % full observation

%% get eigen pairs starts
[v,d]=eig(L); % eigen-decomposition of L
l1=diag(d); % get the eigen-values (make sure that L is symmetric)
l1(1)=0; % set the first eigenvalue to 0
% l12=l1(2); % get the second eigenvalue
% l1m=l1(end); % get the last eigenvalue

% l1=l1/l1m;

% L2=v*d.^2*v'; % Laplacian to the power of 2
% L2=L*L; % Laplacian to the power of 2
% L2=(L2+L2')/2; % make sure that L2 is symmetric
% [v2,d2]=eig(L2);
% l2=diag(d2);
% l2(1)=0;
% l2=l1.^2; %
% l22=l2(2);

% L3=v*d.^3*v'; % Laplacian to the power of 3
% l3=l1.^3;
% L4=v*d.^4*v'; % Laplacian to the power of 4
% l4=l1.^4;
%% get eigen pairs ends
tol_compare=-Inf;
beta_0=ones(1,n_beta); % initial beta's
l_matrix=zeros(n_sample_full,n_beta);
for i=1:n_beta
    l_matrix(:,i)=l1.^i;
end
% beta_0=[1 1 1 1]; % initial beta's

while tol_compare<0
    
    if flag==1
        error_classifier_current=error_classifier;
    end
    
    eig_tol=1e-8;
    lambda=sum(repmat(beta_0,[n_sample_full 1]).*l_matrix,2)+eig_tol;
    lambda=lambda/max(lambda);
%     lambda=lambda/norm(lambda);
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
%         cvx_begin
%         variable x(n_sample_full,1);
%         minimize(x'*cL*x)
% %         minimize(x'*L*x)
%         subject to
%         x(initial_label_index) == class_train_test(initial_label_index);
%         cvx_end
%         x_valid = sign(x);
        %========Graph classifier ends========
        %% Xiaojin
        [fl] = from_plus_minus_to_binary_code(class_train_test, initial_label_index);
        [fu, fu_CMN, q, unlabelled, labelled] = harmonic_function_xiaojin(cL, fl, initial_label_index);
        [fu_CMN_self] = ff_CMN(fu,q,unlabelled);
        current_H = sum( - fu_CMN_self .* log(fu_CMN_self) -  ( 1 - fu_CMN_self ) .* log( 1 - fu_CMN_self ));
        fu_CMN_self_ = [fu_CMN_self 1-fu_CMN_self];
        [x_valid] = from_binary_code_to_plus_minus(fu_CMN_self_, initial_label_index, class_train_test);
        %% KNN see above classifier No. 1
        %% SVM
%         Model_svm_rbf=svm.train(feature_train_test(initial_label_index,:),...
%             class_train_test(initial_label_index),...
%             'kernel_function','rbf','rbf_sigma',1,'autoscale','false');
%         label_svm_rbf=svm.predict(Model_svm_rbf,feature_train_test(~initial_label_index,:));
%         x_valid=class_train_test;
%         x_valid(~initial_label_index)=label_svm_rbf;
        %% Mahalanobis see above classifier No. 2
    end
    
    diff_label = x_valid - class_train_test;
    error_classifier = size(find(diff_label~=0),1)*size(find(diff_label~=0),2)/size(class_test,1);
    
    if flag==0
        disp(['obj before bo : ' num2str(x_valid'*cL*x_valid) ' | n_beta: ' num2str(n_beta) ' | dataset: ' num2str(dataset_i)]);
        disp(['error before bo : ' num2str(error_classifier) ' | n_beta: ' num2str(n_beta) ' | dataset: ' num2str(dataset_i)]);
%         current_obj=x'*cL*x;
        current_obj=current_H;
        error_classifier_0=error_classifier;
        x_valid_previous=x_valid;
    else
        disp(['obj after bo : ' num2str(x_valid'*cL*x_valid) ' | n_beta: ' num2str(n_beta) ' | dataset: ' num2str(dataset_i)]);
        disp(['error after bo : ' num2str(error_classifier) ' | n_beta: ' num2str(n_beta) ' | dataset: ' num2str(dataset_i)]);
%         current_obj=x'*cL*x;
        current_obj=current_H;
        tol_compare=current_obj-previous_obj;
        if x_valid==x_valid_previous %% same classification results
            break
        else
           x_valid_previous=x_valid;
        end
    end
    
    [ beta_0 ] = beta_optimization_LP(n_beta,zeros(1,n_beta),v,l_matrix,x_valid,n_sample_full,eig_tol,tol_set,mo );

    flag=1;
    previous_obj=current_obj;
    
end
disp('===============================================================')
disp(['error w/o bo: ' num2str(error_classifier_0) ' | n_beta: ' num2str(n_beta) ' | dataset: ' num2str(dataset_i)]);
disp(['error with bo: ' num2str(error_classifier_current) ' | n_beta: ' num2str(n_beta) ' | dataset: ' num2str(dataset_i)]);
disp('===============================================================')
end
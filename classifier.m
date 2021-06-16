    function [error_classifier,x_valid] = classifier(dataset_i,n_beta,class_test, ...
    feature_train_test, ...
    initial_label_index, ...
    class_train_test, ...
    classifier_i,...
    M,...
    beta_0_current)

n_sample_full=length(class_train_test);
n_sample_test=length(class_test);
[ class_train_test_,n_class ] = class_num_to_vec( class_train_test,n_sample_full );
[ L ] = graph_Laplacian_train_test( dataset_i,feature_train_test, M ); % full observation
cL=0;
for i_power=1:n_beta
    cL=cL+beta_0_current(i_power)*L^i_power;
end
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
    %%========closed-form version=======
    x_valid_=-pinv(cL(~initial_label_index,~initial_label_index))...
        *cL(~initial_label_index,initial_label_index)...
        *class_train_test_(initial_label_index,:);
    [ x_valid_ ] = class_vec_to_num( x_valid_, n_sample_test );
    x_valid=class_train_test;
%     x_valid(~initial_label_index)=sign(x_valid_);
    x_valid(~initial_label_index)=x_valid_;
    %%==================================
    %% Xiaojin
%     [fl] = from_plus_minus_to_binary_code(class_train_test, initial_label_index);
%     [fu, fu_CMN, q, unlabelled, labelled] = harmonic_function_xiaojin(cL, fl, initial_label_index);
%     [fu_CMN_self] = ff_CMN(fu,q,unlabelled);
%     current_H = sum( - fu_CMN_self .* log(fu_CMN_self) -  ( 1 - fu_CMN_self ) .* log( 1 - fu_CMN_self ));
%     fu_CMN_self_ = [fu_CMN_self 1-fu_CMN_self];
%     [x_valid] = from_binary_code_to_plus_minus(fu_CMN_self_, initial_label_index, class_train_test);
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
end


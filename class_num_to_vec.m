function [ class_train_test_vec,num_of_class ] = class_num_to_vec( class_train_test,n_sample_full )
num_of_class=max(class_train_test);
class_train_test_vec=zeros(n_sample_full,num_of_class);
for i=1:n_sample_full
    class_train_test_vec(i,class_train_test(i))=1;
end
end

